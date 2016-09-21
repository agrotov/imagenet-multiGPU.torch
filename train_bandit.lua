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

--function load_rewards_csv(filePath)
--    -- Read CSV file
--
--    -- Split string
--
--    print("load_rewards_csv")
--
--    function string:split(sep)
--      local sep, fields = sep, {}
--      local pattern = string.format("([^%s]+)", sep)
--      self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
--      return fields
--    end
--
--
--    -- Count number of rows and columns in file
--    local i = 0
--    for line in io.lines(filePath) do
--      if i == 0 then
--        COLS = #line:split(',')
--      end
--      i = i + 1
--    end
--
--    ROWS = i
--   --local ROWS = i - 1  -- Minus 1 because of header
--
--    -- Read data from CSV to tensor
--    local csvFile = io.open(filePath, 'r')
--    local header = csvFile:read()
--
--    local data = torch.Tensor(ROWS, COLS)
--
--    local i = 0
--    for line in csvFile:lines('*l') do
--      i = i + 1
--      local l = line:split(',')
--      for key, val in ipairs(l) do
--        data[i][key] = val
--      end
--    end
--
--    csvFile:close()
--
--
--    return data
--end



function sample_action(model_output, temperature)
    local probabilities_all= probabilities_from_output(model_output, temperature)
    result =  torch.multinomial(probabilities_all,1):long()
    return result
end


function reward_for_actions(loss_matrix, actions, labels)
--    temp = loss_matrix:index(1,actions:view(actions:nElement()))
--    result = temp:gather(2,labels:long():view(labels:nElement(),1))
--    print("actions,labels")
--    print(actions)
--    print("labels")
--    print(labels)
    rewards = (loss_matrix:index(1,actions:view(actions:nElement())):gather(2,labels:long():view(labels:nElement(),1)))
    return  rewards
end

function probabilities_from_output(model_output, temperature)

    local probabilities = torch.exp(model_output)

    if temperature == nil then
        return probabilities
    end

--    print("probabilities",torch.sum(probabilities),torch.mean(probabilities),torch.max(probabilities),torch.min(probabilities),torch.var(probabilities))
--    print("probabilities",torch.sum(torch.exp(probabilities)),torch.mean(torch.exp(probabilities)),torch.max(torch.exp(probabilities)),torch.min(torch.exp(probabilities)),torch.var(torch.exp(probabilities)))

    return probabilities


--    local normalization = torch.sum(torch.exp(probabilities/temperature),2)
--
--    local softmax_probabilities = torch.cdiv(torch.exp(probabilities/temperature),normalization:expandAs(probabilities))

--    return softmax_probabilities

end


function log_probability_of_actions(model_output, actions)
    result =  model_output:gather(2,actions:cuda())
    return result
end


function probability_of_actions(model_output, actions,temperature)
    local probabilities_all = probabilities_from_output(model_output, temperature)
    result =  probabilities_all:gather(2,actions:cuda())
    return result
end

function compute_weight(rewards_arg, probability_actions_student_model, probability_actions_teacher_model)
    local propencity = torch.cdiv(probability_actions_student_model,probability_actions_teacher_model)
    print("propencity", torch.mean(propencity),torch.max(propencity),torch.min(propencity),torch.var(propencity))
    print("probability_actions_student_model",torch.mean(probability_actions_student_model),torch.max(probability_actions_student_model),torch.min(probability_actions_student_model),torch.var(probability_actions_student_model))
    print("probability_actions_teacher_model", torch.mean(probability_actions_teacher_model),torch.max(probability_actions_teacher_model),torch.min(probability_actions_teacher_model),torch.var(probability_actions_teacher_model))
    propencity:clamp(0.01, torch.max(propencity))
    print("propencity clamped", torch.mean(propencity),torch.max(propencity),torch.min(propencity),torch.var(propencity))
    return -torch.cmul(rewards_arg,propencity)
--    return rewards_arg
end

function load_rewards(file_name)
    return csvigo.load({path = file_name, mode = "large"})
end

function compute_target(outputs, size, actions, rewards_arg, probability_actions_student_model, probability_actions_teacher_model, baseline)
    target = torch.Tensor(size):fill(0)
--    print(probability_actions_student_model)
--    print(probability_actions_teacher_model)
--    print(rewards)
--    exit()
--    print(rewards)
    weight = compute_weight(rewards_arg-baseline, probability_actions_student_model, probability_actions_teacher_model)
    log_probability_of_actions_val = log_probability_of_actions(outputs, actions)
--    weight = -torch.cdiv(weight, log_probability_of_actions_val)

--    print("rewards",rewards)
--    print("probability_actions_student_model",probability_actions_student_model)
--    print("probability_actions_teacher_model",probability_actions_teacher_model)
--    print("actions",actions)
    target:scatter(2,actions:long(),weight:float())

--    target = target + 0.5

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





-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()
local outputs = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()



local actions = torch.CudaTensor(opt.batchSize,1)
local rewards= torch.CudaTensor(opt.batchSize,1)
local probabilities_logged= torch.CudaTensor(opt.batchSize,1)



function trainBatch_bandit(inputsCPU, actions_cpu, rewards_cpu, probabilities_logged_cpu, labelsCPU, temperature, batchNumber, baseline)
    model:training()
--    model:evaluate()
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

--    print("rewards_cpu",rewards_cpu)
--    exit()

    local err, target, p_of_actions_student, size_output


    feval = function(x)
        model:zeroGradParameters()

        print("gradParameters",torch.mean(gradParameters))


        outputs = model:forward(inputs)

        print("outputs", torch.mean(outputs),torch.min(outputs),torch.max(outputs))

        size_output = outputs:size()
        p_of_actions_student = probability_of_actions(outputs, actions, temperature)

        --print(torch.mean(p_of_actions_student), torch.mean(probabilities_logged))
--        rewards_fake = torch.rand(p_of_actions_student:size()):cuda()

        target = compute_target(outputs, size_output,actions, rewards, p_of_actions_student, probabilities_logged, baseline)

        gpu_target = target:cuda()

--        print("target",torch.mean(target),torch.max(torch.abs(target)),torch.min(torch.abs(target)))



--        err = rewards:mean()
        --print("target",target)
        model:backward(inputs, gpu_target)

        nan_mask = gradParameters:ne(gradParameters)
        non_nan_mask = gradParameters:eq(gradParameters)
        print("sum nan ",torch.sum(nan_mask),torch.sum(non_nan_mask))

        print("gradParameters",torch.mean(gradParameters[non_nan_mask]),torch.max(gradParameters[non_nan_mask]),torch.min(gradParameters[non_nan_mask]))
--        gradParameters:clamp(-5, 5)
--        print("gradParameters",torch.mean(gradParameters),torch.max(gradParameters),torch.min(gradParameters))

        gradParameters:clamp(-5, 5)

        return err, gradParameters
    end
    print("optimState",optimState)
    optim.sgd(feval, parameters, optimState)

    -- DataParallelTable's syncParameters
    if model.needsSync then
        model:syncParameters()
    end


    cutorch.synchronize()

    model:evaluate()

    print("parameters",torch.mean(parameters),torch.max(parameters),torch.min(parameters))
    outputs = model:forward(inputs)

    print("outputs new", torch.mean(outputs),torch.min(outputs),torch.max(outputs))

    p_of_actions_student_new = probability_of_actions(outputs, actions, temperature)
--    print(torch.cat(rewards,torch.cat(torch.cat(probabilities_logged,p_of_actions_student_new,2),p_of_actions_student_new-p_of_actions_student,2),2))
--    print(rewards)

    rewards_sum_logged = torch.sum(torch.cmul(rewards,probabilities_logged))/torch.sum(probabilities_logged)
    rewards_sum_old = torch.sum(torch.cmul(rewards,p_of_actions_student))/torch.sum(p_of_actions_student)
    rewards_sum_new = torch.sum(torch.cmul(rewards,p_of_actions_student_new))/torch.sum(p_of_actions_student_new)
--    print("p_of_actions_student_new",p_of_actions_student_new[p_of_actions_student_new:gt(0.5)]:size())
    print("Probabilities", torch.mean(probabilities_logged),torch.mean(p_of_actions_student),torch.mean(p_of_actions_student_new))
    print("Rewards",rewards_sum_logged,rewards_sum_old,rewards_sum_new ,rewards_sum_new -rewards_sum_old, torch.mean(p_of_actions_student_new-p_of_actions_student))


    --   print(p_of_actions_student)

    --    top-1 error
--    if batchNumber % 10 == 0 then
--        full_information_full_test(inputsCPU, actions_cpu, rewards_cpu, probabilities_logged_cpu, labelsCPU, temperature, batchNumber, baseline)
        full_information_test(inputsCPU, actions, rewards, probabilities_logged, labelsCPU, p_of_actions_student_new, batchNumber)
--        full_information_test(inputsCPU, labelsCPU, batchNumber, rewards_cpu, probabilities_logged)
--    end
    dataTimer:reset()

    return outputs
end


--function full_information_full_test(inputsCPU, actions_cpu, rewards_cpu, probabilities_logged_cpu, labelsCPU, temperature, batchNumber, baseline)
function full_information_test(inputsCPU, actions, rewards_logged, probabilities_logged, labelsCPU, p_of_actions_student_new, batchNumber)
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    model:evaluate()
    local top1_epoch = 0
    local top1 = 0
    local actions_eva = torch.LongTensor(opt.batchSize)
    local rewards_model = 0
    outputs = model:forward(inputs)



    print("outputs test", torch.mean(outputs),torch.min(outputs),torch.max(outputs))


    local _,prediction_sorted = outputs:float():sort(2, true) -- descending
    for i=1,opt.batchSize do
        if prediction_sorted[i][1] == labelsCPU[i] then
            --        if actions[i] == labelsCPU[i] then
            top1_epoch = top1_epoch + 1;
            top1 = top1 + 1
        end
        actions_eva[i] = prediction_sorted[i][1]
    end
    top1 = top1 * 100 / opt.batchSize;

--    print("actions_eva",actions_eva)

--    local actions_sample = sample_action(outputs,1)

    rewards_eva = 1-reward_for_actions(loss_matrix, actions_eva, labelsCPU)

    diff_rewards = rewards_eva:mean() - rewards_logged:mean()

    rewards_sum_logged = torch.sum(torch.cmul(rewards_logged,probabilities_logged))/torch.sum(probabilities_logged)
    rewards_sum_new = torch.sum(torch.cmul(rewards_logged,p_of_actions_student_new))/torch.sum(p_of_actions_student_new)


    print("rewards, new_probabilities",torch.cat(rewards,new_probabilities,2))

    -- Calculate top-1 error, and print information
    print(('Epoch: [%d][%d/%d]\tTime %.3f Reward %.4f RewardsLogged %.4f RewardDiff %.4f  WeightedRewards %.4f WeightedRewardsNew %.4f WeightedRewardsDiff %.4f Top1-%%: %.2f LR %.0e'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real,rewards_eva:mean(), rewards:mean(),  diff_rewards,rewards_sum_logged, rewards_sum_new, rewards_sum_new - rewards_sum_logged, top1,
        optimState.learningRate))
end


function full_information_full_test(inputsCPU, actions_cpu, rewards_cpu, probabilities_logged_cpu, labelsCPU, temperature, batchNumber, baseline)
    model:evaluate()
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

--    print("rewards_cpu",rewards_cpu)
--    exit()

    local err, target, p_of_actions_student, size_output

    model:evaluate()
    local top1_epoch = 0
    local top1 = 0
    local actions_eva = torch.LongTensor(opt.batchSize,1)
    local rewards_model = 0
    outputs = model:forward(inputs)

    new_probabilities = probability_of_actions(outputs, actions, temperature)

    local _,prediction_sorted = outputs:float():sort(2, true) -- descending
    for i=1,opt.batchSize do
        if prediction_sorted[i][1] == labelsCPU[i] then
            --        if actions[i] == labelsCPU[i] then
            top1_epoch = top1_epoch + 1;
            top1 = top1 + 1
        end
        actions_eva[i] = prediction_sorted[i][1]
    end
    top1 = top1 * 100 / opt.batchSize;


    rewards_eva = 1-reward_for_actions(loss_matrix, actions_eva, labelsCPU)


    diff_rewards = rewards_eva:mean() - rewards:mean()

--    print("loss_matrix",loss_matrix)


    rewards_sum_logged = torch.sum(torch.cmul(rewards,probabilities_logged))/torch.sum(probabilities_logged)
    rewards_sum_new = torch.sum(torch.cmul(rewards,new_probabilities))/torch.sum(new_probabilities)

    -- Calculate top-1 error, and print information
    print(('Epoch: [%d][%d/%d]\tTime %.3f Reward %.4f RewardsLogged %.4f RewardDiff %.4f  WeightedRewards %.4f WeightedRewardsNew %.4f WeightedRewardsDiff %.4f Top1-%%: %.2f LR %.0e'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real,rewards_eva:mean(), rewards:mean(),  diff_rewards,rewards_sum_logged, rewards_sum_new, rewards_sum_new - rewards_sum_logged, top1,
        optimState.learningRate))

end



