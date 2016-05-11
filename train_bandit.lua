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
require("os")


function load_rewards_mnist()
    print(torch.eye(10))
    return torch.eye(10)
end

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



function sample_action(model_output, temperature)
    local probabilities_all= probabilities_from_output(model_output, temperature)
    result =  torch.multinomial(probabilities_all,1):long()
    for i=1,result:size()[1] do
--        print(result[i])
        if result[i]:eq(0) then
            print(i)
--            print(probabilities_all[i])
        end

    end

    print("probabilities_all:size()")
--    print(probabilities_all)
    print("actions")
    print(result)
    return result
end


function reward_for_actions(loss_matrix, actions, labels)
--    temp = loss_matrix:index(1,actions:view(actions:nElement()))
--    result = temp:gather(2,labels:long():view(labels:nElement(),1))
--    print("actions")
--    print(actions)
    rewards = (loss_matrix:index(1,actions:view(actions:nElement())):gather(2,labels:long():view(labels:nElement(),1)))
    return  rewards
end

function probabilities_from_output(model_output, temperature)

    local probabilities = torch.exp(model_output)

    if temperature == nil then
        return probabilities
    end


    local normalization = torch.sum(torch.exp(probabilities/temperature),2)

    local softmax_probabilities = torch.cdiv(torch.exp(probabilities/temperature),normalization:expandAs(probabilities))

    return softmax_probabilities

end

function probability_of_actions(model_output, actions,temperature)
    local probabilities_all = probabilities_from_output(model_output, temperature)
    result =  probabilities_all:gather(2,actions:cuda())
    return result
end

function compute_weight(rewards, probability_actions_student_model, probability_actions_teacher_model)
    return -torch.cmul(rewards,torch.cdiv(probability_actions_student_model,probability_actions_teacher_model))
end

function load_rewards(file_name)
    return csvigo.load({path = file_name, mode = "large"})
end

function compute_target(size, actions, rewards, probability_actions_student_model, probability_actions_teacher_model)
    target = torch.Tensor(size):fill(0)
--    print(probability_actions_student_model)
--    print(probability_actions_teacher_model)
--    print(rewards)
--    exit()
    weight = compute_weight(rewards, probability_actions_student_model, probability_actions_teacher_model)


    target:scatter(2,actions:long(),weight:float())

    return target
end




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


if opt.optimState ~= nil and opt.optimState ~= 'none' then
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

   loss_matrix = load_rewards_csv("/home/agrotov/imagenet-multiGPU.torch/loss_matrix.txt")

   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0

--   opt.epochSize = 1

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
local outputs = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU, optimState)

   batchNumber = batchNumber or 1

   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local err, target, p_of_actions_teacher, p_of_actions_student, rewards, gpu_target, actions, size_output


   feval = function(x)
      model:zeroGradParameters()
      outputs = model:forward(inputs)
      size_output = outputs:size()
      actions = sample_action(outputs)

      p_of_actions_teacher = probability_of_actions(outputs, actions)
      p_of_actions_student = probability_of_actions(outputs, actions)
      rewards = reward_for_actions(loss_matrix, actions, labels)
      target = compute_target(size_output,actions, rewards, p_of_actions_student, p_of_actions_teacher)

      gpu_target = target:cuda()


      err = rewards:mean()
      model:backward(inputs, gpu_target)
      return err, gradParameters
   end

   optim.sgd(feval, parameters, optimState)

   -- DataParallelTable's syncParameters
   if model.needsSync then
      model:syncParameters()
   end
   

   cutorch.synchronize()
   batchNumber = batchNumber + 1
--   loss_epoch = loss_epoch + err
   -- top-1 error
--   local top1 = 0
--   do
--      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
--      for i=1,opt.batchSize do
--	 if prediction_sorted[i][1] == labelsCPU[i] then
--	    top1_epoch = top1_epoch + 1;
--	    top1 = top1 + 1
--	 end
--      end
--      top1 = top1 * 100 / opt.batchSize;
--   end
--   -- Calculate top-1 error, and print information
--   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DataLoadingTime %.3f'):format(
--          epoch, batchNumber, opt.epochSize, timer:time().real, err, top1,
--          optimState.learningRate, dataLoadingTime))

   dataTimer:reset()

--   print(outputs)
--   if batchNumber > 3000 then
--       exit()
--    end
   return outputs
end


local actions = torch.CudaTensor(opt.batchSize,1)
local rewards= torch.CudaTensor(opt.batchSize,1)
local probabilities_logged= torch.CudaTensor(opt.batchSize,1)


function trainBatch_bandit(inputsCPU, actions_cpu, rewards_cpu, probabilities_logged_cpu, optimState, labelsCPU)

   batchNumber = batchNumber or 1

   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   actions:copy(actions_cpu)
   rewards:copy(rewards_cpu)
   probabilities_logged:copy(probabilities_logged_cpu)


   local err, target, p_of_actions_student, size_output


   feval = function(x)
      model:zeroGradParameters()
      outputs = model:forward(inputs)
      size_output = outputs:size()
      p_of_actions_student = probability_of_actions(outputs, actions)
      target = compute_target(size_output,actions, rewards, p_of_actions_student, probabilities_logged)

      gpu_target = target:cuda()



      err = rewards:mean()
      model:backward(inputs, gpu_target)
      return err, gradParameters
   end

   optim.sgd(feval, parameters, optimState)

   -- DataParallelTable's syncParameters
   if model.needsSync then
      model:syncParameters()
   end


   cutorch.synchronize()

   print(p_of_actions_student)

--    top-1 error
   local top1_epoch = 0
   local top1 = 0
   do
      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      for i=1,opt.batchSize do
         if prediction_sorted[i][1] == labelsCPU[i] then
--        if actions[i] == labelsCPU[i] then
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

   return outputs
end
