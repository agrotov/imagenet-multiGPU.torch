--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require("csvigo")
autograd = require 'autograd'





function load_rewards_csv(filePath)
    -- Read CSV file

    -- Split string

    print("load_rewards_csv")

    function string:split(sep)
      local sep, fields = sep, {}
      local pattern = string.format("([^%s]+)", sep)
      self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
      return fields
    end


    -- Count number of rows and columns in file
    local i = 0
    for line in io.lines(filePath) do
      if i == 0 then
        COLS = #line:split(',')
      end
      i = i + 1
    end

    ROWS = i
   --local ROWS = i - 1  -- Minus 1 because of header

    -- Read data from CSV to tensor
    local csvFile = io.open(filePath, 'r')
    local header = csvFile:read()

    local data = torch.Tensor(ROWS, COLS)

    local i = 0
    for line in csvFile:lines('*l') do
      i = i + 1
      local l = line:split(',')
      for key, val in ipairs(l) do
        data[i][key] = val
      end
    end

    csvFile:close()


    return data
end



local function sample_action(model_output)
    return torch.multinomial(model_output,1):long()
end


local function reward_for_actions(loss_matrix, actions, labels)
--    temp = loss_matrix:index(1,actions:view(actions:nElement()))
--    result = temp:gather(2,labels:long():view(labels:nElement(),1))
    return  loss_matrix:index(1,actions:view(actions:nElement())):gather(2,labels:long():view(labels:nElement(),1))
end

local function probability_of_actions(model_output, actions)
    return model_output:float():gather(2,actions)
end

local function compute_weight(rewards, probability_actions_student_model, probability_actions_teacher_model)
    return torch.cdiv(rewards,torch.cmul(probability_actions_student_model,probability_actions_teacher_model))
end

local function load_rewards(file_name)
    return csvigo.load({path = file_name, mode = "large"})
end

local function compute_target(size, actions, rewards, probability_actions_student_model, probability_actions_teacher_model)
    target = torch.Tensor(size)

--    print("target")
--    print(target:size())
--    pring(target:type())

    weight = compute_weight(rewards, probability_actions_student_model, probability_actions_teacher_model)

--    print("weight")
--    print(weight:size())
--    pring(weight:type())

    target:scatter(2,actions,weight)

--    print("target:scatter")
--    print(target:size())
--    pring(target:type())


    return target
end


------------------------------------------------------------------------------
-- Custom AutoGrad MSE Criterion Loss Function designed to over/under predict.
------------------------------------------------------------------------------
local autoMaximizationCriterion = function(x, y)
    print ("autoMaximizationCriterion x")
    print(x)
--    print("x:type()")
--    print(x:type())
    print("x:size()")
    print(x:size())
    print(x)
    print("x done")
    return torch.sum(torch.cmul(x,y))
end


-- Create an autograd criterion using the loss function above.
bandit_criterion = autograd.nn.AutoCriterion('AutoMax')(autoMaximizationCriterion)



--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   1e-2,   5e-4, },
        { 19,     29,   5e-3,   5e-4  },
        { 30,     43,   1e-3,   0 },
        { 44,     52,   5e-4,   0 },
        { 53,    1e8,   1e-4,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

--   modelf, params_f = autograd.functionalize(model)

   loss_matrix = load_rewards_csv("/home/agrotov/imagenet-multiGPU.torch/loss_matrix.txt")

   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0

   opt.epochSize = 1

   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels = trainLoader:sample(opt.batchSize)
            return inputs, labels
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   model:clearState()
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()







-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

--   local err, outputs
   feval = function(x)
      model:zeroGradParameters()
      outputs = model:forward(inputs)


--      err = criterion:forward(outputs, labels)
--      local gradOutputs = criterion:backward(outputs, labels)
--      model:backward(inputs, gradOutputs)

      local size_output = outputs:size()


      local actions = sample_action(outputs)
      local p_of_actions_teacher = probability_of_actions(outputs, actions)
      local p_of_actions_student = probability_of_actions(outputs, actions)
      local rewards = reward_for_actions(loss_matrix, actions, labels)
      target = compute_target(size_output,actions, rewards, p_of_actions_student, p_of_actions_teacher)
--      local cuda_target = torch.CudaTensor(target)

--      print("target")
--      print(target:size())
--      print(target:type())
--      print("outputs before")
--      print(outputs:size())
--      print(outputs:type())

      err = criterion:forward(outputs, labels)
--      err = 0
--      local gradOutputs = criterion:backward(outputs, target)
--      grads_ones = torch.Tensor(outputs:size())
--      s = grads_ones:storage()
--      for i=1,s:size() do -- fill up the Storage
--        s[i] = 1
--      end

      model:backward(inputs, gradOutputs)



      return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)

   -- DataParallelTable's syncParameters
   if model.needsSync then
      model:syncParameters()
   end
   

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   -- top-1 error
   local top1 = 0
   do
      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      for i=1,opt.batchSize do
	 if prediction_sorted[i][1] == labelsCPU[i] then
	    top1_epoch = top1_epoch + 1;
	    top1 = top1 + 1
	 end
      end
      top1 = top1 * 100 / opt.batchSize;
   end
   -- Calculate top-1 error, and print information
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, err, top1,
          optimState.learningRate, dataLoadingTime))

   dataTimer:reset()
end
