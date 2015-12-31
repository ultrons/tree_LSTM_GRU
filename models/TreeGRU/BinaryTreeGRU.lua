--[[

  A Binary Tree-GRU with input at the leaf nodes.

--]]

local BinaryTreeGRU, parent = torch.class('treelstm.BinaryTreeGRU', 'treelstm.TreeGRU')

function BinaryTreeGRU:__init(config)
  parent.__init(self, config)
  self.gate_output = config.gate_output
  if self.gate_output == nil then self.gate_output = true end

  -- a function that instantiates an output module that takes the hidden state h as input
  self.output_module_fn = config.output_module_fn
  self.criterion = config.criterion

  -- leaf input module
  self.leaf_module = self:new_leaf_module()
  self.leaf_modules = {}

  -- composition module
  self.composer = self:new_composer()
  self.composers = {}

  -- output module
  self.output_module = self:new_output_module()
  self.output_modules = {}

  -- bias for forget gate
  self.bias = config.bias
end

function BinaryTreeGRU:new_leaf_module()
  local input = nn.Identity()()
  local z = nn.Sigmoid()(nn.Linear(self.in_dim, self.mem_dim)(input))
  local r = nn.Linear(self.in_dim, self.mem_dim)(input)
  local h = nn.CMulTable(){nn.Tanh()(r), z}

  local leaf_module = nn.gModule({input}, {h})
  if self.leaf_module ~= nil then
    share_params(leaf_module, self.leaf_module)
  end
  return leaf_module
end

function BinaryTreeGRU:new_composer()
  local lh = nn.Identity()()
  local rh = nn.Identity()()

  local new_gate_z = function()
    return nn.CAddTable(){
      nn.Linear(self.mem_dim, self.mem_dim)(nn.Tanh()(lh)),
      nn.Linear(self.mem_dim, self.mem_dim)(nn.Tanh()(rh))
    }
  end
  local new_gate_r = function()
    return nn.CAddTable(){
      nn.Linear(self.mem_dim, self.mem_dim)(lh):annotate{name = 'resetGate'},
      nn.Linear(self.mem_dim, self.mem_dim)(rh):annotate{name = 'resetGate'}
    }
  end

  local z = nn.Sigmoid()(new_gate_z())
  local r = nn.Sigmoid()(new_gate_r())

  local new_gate_h = function()
    return nn.CAddTable(){
      nn.Linear(self.mem_dim, self.mem_dim)(nn.CMulTable(){lh, r}),
      nn.Linear(self.mem_dim, self.mem_dim)(nn.CMulTable(){rh, r})
    }
  end

  local h = nn.CAddTable(){
      nn.CMulTable(){nn.Tanh()(new_gate_h()), z},
      nn.CMulTable(){nn.AddConstant(1,false)(nn.MulConstant(-1,false)(z)), lh},
      nn.CMulTable(){nn.AddConstant(1,false)(nn.MulConstant(-1,false)(z)), rh}
    }

  local composer = nn.gModule({lh, rh}, {h})
  if self.composer ~= nil then
    share_params(composer, self.composer)
  end
  return composer
end

function BinaryTreeGRU:new_output_module()
  if self.output_module_fn == nil then return nil end
  local output_module = self.output_module_fn()
  if self.output_module ~= nil then
    share_params(output_module, self.output_module)
  end
  return output_module
end

function BinaryTreeGRU:forward(tree, inputs)
  local lloss, rloss = 0, 0
  if tree.num_children == 0 then
    self:allocate_module(tree, 'leaf_module')
    tree.state = tree.leaf_module:forward(inputs[tree.leaf_idx])
  else
    self:allocate_module(tree, 'composer')
    -- Set bias
    if self.bias ~= nil then
      for _,node in ipairs(tree.composer.forwardnodes) do
        if node.data.annotations.name == "resetGate" then
            node.data.module.bias:fill(self.bias)
        end
      end
    end
    -- get child hidden states
    local lvecs, lloss = self:forward(tree.children[1], inputs)
    local rvecs, rloss = self:forward(tree.children[2], inputs)
    local lh = self:unpack_state(lvecs)
    local rh = self:unpack_state(rvecs)
    
    -- compute state and output
    tree.state = tree.composer:forward({lh, rh})
  end

  local loss
  if self.output_module ~= nil then
    self:allocate_module(tree, 'output_module')
    tree.output = tree.output_module:forward(tree.state)
    if self.train then
      loss = self.criterion:forward(tree.output, tree.gold_label) + lloss + rloss
    end
  end
  return tree.state, loss
end

function BinaryTreeGRU:backward(tree, inputs, grad)
  local grad_inputs = torch.Tensor(inputs:size())
  self:_backward(tree, inputs, grad, grad_inputs)
  return grad_inputs
end

function BinaryTreeGRU:_backward(tree, inputs, grad, grad_inputs)
  local output_grad = self.mem_zeros
  if tree.output ~= nil and tree.gold_label ~= nil then
    output_grad = tree.output_module:backward(
      tree.state, self.criterion:backward(tree.output, tree.gold_label))
  end
  self:free_module(tree, 'output_module')
  if tree.num_children == 0 then
    grad_inputs[tree.leaf_idx] = tree.leaf_module:backward(
      inputs[tree.leaf_idx],
      grad + output_grad)
    self:free_module(tree, 'leaf_module')
  else
    local lh, rh = self:get_child_states(tree)
    local composer_grad = tree.composer:backward(
      {lh, rh},
      grad + output_grad)
    self:free_module(tree, 'composer')
    self:_backward(tree.children[1], inputs, composer_grad[1], grad_inputs)
    self:_backward(tree.children[2], inputs, composer_grad[2], grad_inputs)
  end
  tree.state = nil
  tree.output = nil
end

function BinaryTreeGRU:parameters()
  local params, grad_params = {}, {}
  local cp, cg = self.composer:parameters()
  tablex.insertvalues(params, cp)
  tablex.insertvalues(grad_params, cg)
  local lp, lg = self.leaf_module:parameters()
  tablex.insertvalues(params, lp)
  tablex.insertvalues(grad_params, lg)
  if self.output_module ~= nil then
    local op, og = self.output_module:parameters()
    tablex.insertvalues(params, op)
    tablex.insertvalues(grad_params, og)
  end
  return params, grad_params
end

--
-- helper functions
--

function BinaryTreeGRU:unpack_state(state)
  local h
  if state == nil then
    h = self.mem_zeros
  else
    h = state
  end
  return h
end

function BinaryTreeGRU:get_child_states(tree)
  local lh, rh
  if tree.children[1] ~= nil then
    lh = self:unpack_state(tree.children[1].state)
  end

  if tree.children[2] ~= nil then
    rh = self:unpack_state(tree.children[2].state)
  end
  return lh, rh
end

function BinaryTreeGRU:clean(tree)
  tree.state = nil
  tree.output = nil
  self:free_module(tree, 'leaf_module')
  self:free_module(tree, 'composer')
  self:free_module(tree, 'output_module')
  for i = 1, tree.num_children do
    self:clean(tree.children[i])
  end
end
