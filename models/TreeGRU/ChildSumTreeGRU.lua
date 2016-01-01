--[[

  A Child-Sum Tree-GRU with input at each node.

--]]

local ChildSumTreeGRU, parent = torch.class('treelstm.ChildSumTreeGRU', 'treelstm.TreeGRU')

function ChildSumTreeGRU:__init(config)
  parent.__init(self, config)
  self.gate_output = config.gate_output
  if self.gate_output == nil then self.gate_output = true end

  -- a function that instantiates an output module that takes the hidden state h as input
  self.output_module_fn = config.output_module_fn
  self.criterion = config.criterion

  -- composition module
  self.composer = self:new_composer()
  self.composers = {}

  -- output module
  self.output_module = self:new_output_module()
  self.output_modules = {}
end

function ChildSumTreeGRU:new_composer()
  local input = nn.Identity()()
  local child_h = nn.Identity()()
  local child_h_sum = nn.Sum(1)(child_h)

  local z = nn.Sigmoid()(
    nn.CAddTable(){
      nn.Linear(self.in_dim, self.mem_dim)(input),
      nn.Linear(self.mem_dim, self.mem_dim)(nn.Tanh()(child_h_sum))
    })

  local r = nn.Sigmoid()(
    treelstm.CRowAddTable(){
      nn.TemporalConvolution(self.mem_dim, self.mem_dim, 1)(child_h):annotate{name = 'resetGate'},
      nn.Linear(self.in_dim, self.mem_dim)(input):annotate{name = 'resetGate'},
    })

  local reset_child_tanh = nn.Tanh()(
    nn.CAddTable(){
      nn.Sum(1)(nn.CMulTable(){r, child_h}),
      nn.Linear(self.in_dim, self.mem_dim)(input),
    })

  local h = nn.CAddTable(){
      nn.CMulTable(){reset_child_tanh, z},
      nn.CMulTable(){nn.AddConstant(1,false)(nn.MulConstant(-1,false)(z)), child_h_sum},
    }

  local composer = nn.gModule({input, child_h}, {h})
  if self.composer ~= nil then
    share_params(composer, self.composer)
  end
  return composer
end

function ChildSumTreeGRU:new_output_module()
  if self.output_module_fn == nil then return nil end
  local output_module = self.output_module_fn()
  if self.output_module ~= nil then
    share_params(output_module, self.output_module)
  end
  return output_module
end

function ChildSumTreeGRU:forward(tree, inputs)
  local loss = 0
  for i = 1, tree.num_children do
    local _, child_loss = self:forward(tree.children[i], inputs)
    loss = loss + child_loss
  end
  local child_h = self:get_child_states(tree)
  self:allocate_module(tree, 'composer')
  -- Set bias
  if self.bias ~= nil then
    for _,node in ipairs(tree.composer.forwardnodes) do
      if node.data.annotations.name == "resetGate" then
          node.data.module.bias:fill(self.bias)
      end
    end
  end
  tree.state = tree.composer:forward{inputs[tree.idx], child_h}

  if self.output_module ~= nil then
    self:allocate_module(tree, 'output_module')
    tree.output = tree.output_module:forward(tree.state)
    if self.train and tree.gold_label ~= nil then
      loss = loss + self.criterion:forward(tree.output, tree.gold_label)
    end
  end
  return tree.state, loss
end

function ChildSumTreeGRU:backward(tree, inputs, grad)
  local grad_inputs = torch.Tensor(inputs:size())
  self:_backward(tree, inputs, grad, grad_inputs)
  return grad_inputs
end

function ChildSumTreeGRU:_backward(tree, inputs, grad, grad_inputs)
  local output_grad = self.mem_zeros
  if tree.output ~= nil and tree.gold_label ~= nil then
    output_grad = tree.output_module:backward(
      tree.state, self.criterion:backward(tree.output, tree.gold_label))
  end
  self:free_module(tree, 'output_module')
  tree.output = nil

  local child_h = self:get_child_states(tree)
  local composer_grad = tree.composer:backward(
    {inputs[tree.idx], child_h},
    grad + output_grad)
  self:free_module(tree, 'composer')
  tree.state = nil
  grad_inputs[tree.idx] = composer_grad[1]
  local child_h_grads = composer_grad[2]
  for i = 1, tree.num_children do
    self:_backward(tree.children[i], inputs, child_h_grads[i], grad_inputs)
  end
end

function ChildSumTreeGRU:clean(tree)
  self:free_module(tree, 'composer')
  self:free_module(tree, 'output_module')
  tree.state = nil
  tree.output = nil
  for i = 1, tree.num_children do
    self:clean(tree.children[i])
  end
end

function ChildSumTreeGRU:parameters()
  local params, grad_params = {}, {}
  local cp, cg = self.composer:parameters()
  tablex.insertvalues(params, cp)
  tablex.insertvalues(grad_params, cg)
  if self.output_module ~= nil then
    local op, og = self.output_module:parameters()
    tablex.insertvalues(params, op)
    tablex.insertvalues(grad_params, og)
  end
  return params, grad_params
end

function ChildSumTreeGRU:get_child_states(tree)
  local child_h
  if tree.num_children == 0 then
    child_h = torch.zeros(1, self.mem_dim)
  else
    child_h = torch.Tensor(tree.num_children, self.mem_dim)
    for i = 1, tree.num_children do
       child_h[i] = tree.children[i].state
    end
  end
  return child_h
end
