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



function sample_action(model_output, temperature)
    local probabilities_all= probabilities_from_output(model_output, temperature)
    result =  torch.multinomial(probabilities_all,1):long()
    return result
end


function reward_for_actions(loss_matrix, actions, labels)
--    rewards = (loss_matrix:index(1,actions:view(actions:nElement())):gather(2,labels:long():view(labels:nElement(),1)))

    rewards_zero_one = torch.Tensor(actions:size())
    for i=1,actions:size()[1] do
        if actions[i][1] == labels[i] then
            rewards_zero_one[i] = 0
        else
            rewards_zero_one[i] = 1
        end
    end

    return  rewards_zero_one
end

function probabilities_from_output(model_output, temperature)

    local probabilities = torch.exp(model_output)

    if temperature == nil then
        return probabilities
    end
    return probabilities

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
    propencity:clamp(0.01, torch.max(propencity))
    return -torch.cmul(rewards_arg,propencity)
--    return rewards_arg
end

function load_rewards(file_name)
    return csvigo.load({path = file_name, mode = "large"})
end

nuber_of_data_processed =0
mean_so_far = 0
m2_value = 0

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


function compute_variance_batch(inputsCPU, actions_cpu, rewards_cpu, temperature)
    print("compute_variance_batch")
    model:training()

    cutorch.synchronize()
    collectgarbage()
    -- transfer over to GPU
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    actions:copy(actions_cpu)
    rewards:copy(rewards_cpu)

    outputs = model:forward(inputs)

    p_of_actions_student = probability_of_actions(outputs, actions, temperature)

    weighted_reward = torch.cmul(rewards,p_of_actions_student)

    print("weighted_reward",weighted_reward)

    for i=1,opt.batchSize do
        nuber_of_data_processed = nuber_of_data_processed + 1
        print("nuber_of_data_processed",nuber_of_data_processed)
        print("mean_so_far",mean_so_far)
        print("weighted_reward[i]",weighted_reward[i])
        delta = weighted_reward[i] - mean_so_far
--        print("delta",delta)
--        mean_so_far = mean_so_far + delta/nuber_of_data_processed
--        m2_value = m2_value + delta*(weighted_reward[i] - mean_so_far)
    end

--    print("nuber_of_data_processed",nuber_of_data_processed)
--    print("mean_so_far",mean_so_far)
--    print("m2_value",m2_value)

end



function compute_variance_constants(inputsCPU, actions_cpu, rewards_cpu, probabilities_logged_cpu, labelsCPU, temperature, baseline)
    model:training()

    cutorch.synchronize()
    collectgarbage()
    timer:reset()

    -- transfer over to GPU
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    actions:copy(actions_cpu)
    rewards:copy(rewards_cpu)
    probabilities_logged:copy(probabilities_logged_cpu)

    local err, target, p_of_actions_student, size_output
    outputs = model:forward(inputs)
    size_output = outputs:size()
    p_of_actions_student = probability_of_actions(outputs, actions, temperature)
    target = compute_target(outputs, size_output,actions, rewards, p_of_actions_student, probabilities_logged, baseline)


end

function get_variance_gradient(rewards_arg,probability_actions)
    weighted_rewards_of_logging_policy = torch.cmul(rewards_arg,probability_actions)
    mean_weighted_rewards_of_logging_policy  = weighted_rewards_of_logging_policy:mean()
    diff_mean_weighted_rewards_of_logging_policy = (weighted_rewards_of_logging_policy-mean_weighted_rewards_of_logging_policy)
    diff_mean_weighted_rewards_of_logging_policy_square = torch.cmul(diff_mean_weighted_rewards_of_logging_policy,diff_mean_weighted_rewards_of_logging_policy)
    diff_mean_weighted_rewards_of_logging_policy_square_sum = torch.sum(diff_mean_weighted_rewards_of_logging_policy_square)
    variance_of_logging_policy = diff_mean_weighted_rewards_of_logging_policy_square_sum / (opt.batchSize - 1)
    sqrt_variance_of_logging_policy = torch.sqrt(variance_of_logging_policy)

    A_w0 = - mean_weighted_rewards_of_logging_policy / ((opt.batchSize - 1) * sqrt_variance_of_logging_policy)
    b_w0 = 1/(2 * (opt.batchSize - 1) * sqrt_variance_of_logging_policy)

    print("A_w0",A_w0)
    print("b_w0",b_w0)

    return A_w0,b_w0
end


function compute_target(outputs, size, actions, rewards_arg, probability_actions_student_model, probability_actions_teacher_model, baseline)
    target = torch.Tensor(size):fill(0)
    weight = compute_weight(rewards_arg-opt.baseline, probability_actions_student_model, probability_actions_teacher_model)
    log_probability_of_actions_val = log_probability_of_actions(outputs, actions)

    weight = -torch.cdiv(weight, log_probability_of_actions_val)
    target:scatter(2,actions:long(),weight:float())

    expected_reward = torch.cmul(probability_actions_student_model,rewards_arg-opt.baseline)
    expected_reward_scattered = torch.Tensor(size):fill(0)
    expected_reward_scattered:scatter(2,actions:long(),expected_reward:float())

--    variance_grad = get_variance_gradient(rewards_arg,probability_actions_teacher_model, expected_reward_scattered, target)

--    return target + opt.variance_reg * variance_grad
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




-------------------------------------------------------------------------------------------

rewards_sum_new_train,rewards_sum_logged_train,rewards_new_mean_train, rewards_logged_mean_train = 0
num_batches = 0

function trainBatch_bandit(inputsCPU, actions_cpu, rewards_cpu, probabilities_logged_cpu, labelsCPU, temperature, baseline)


    model:training()

    cutorch.synchronize()
    collectgarbage()
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
        p_of_actions_student = probability_of_actions(outputs, actions, temperature)

        target = compute_target(outputs, size_output,actions, rewards, p_of_actions_student, probabilities_logged, baseline)

        gpu_target = target:cuda()
        model:backward(inputs, gpu_target)

--        nan_mask = gradParameters:ne(gradParameters)
--        non_nan_mask = gradParameters:eq(gradParameters)
--        print("sum nan ",torch.sum(nan_mask),torch.sum(non_nan_mask))
        print("gradParameters", gradParameters:mean(),gradParameters:min(),gradParameters:max())
        print("parameters", parameters:mean(),parameters:min(),parameters:max())
        gradParameters:clamp(-5, 5)

        return err, gradParameters
    end
    optim.sgd(feval, parameters, optimState)

    -- DataParallelTable's syncParameters
    if model.needsSync then
        model:syncParameters()
    end

    rewards_sum_new_train,rewards_sum_logged_train,rewards_new_mean_train, rewards_logged_mean_train = full_information_full_test(inputsCPU, actions, rewards, probabilities_logged, labelsCPU, temperature)

    dataTimer:reset()


end



function full_information_full_test(inputsCPU, actions_cpu, rewards_cpu, probabilities_logged_cpu, labelsCPU, temperature)
    batch_number = batch_number + 1

    model:evaluate()

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


    rewards_sum_logged = torch.sum(torch.cmul(rewards,probabilities_logged))/torch.sum(probabilities_logged)
    rewards_sum_new = torch.sum(torch.cmul(rewards,new_probabilities))/torch.sum(new_probabilities)

    print("outputs", outputs:mean(),outputs:min(),outputs:max())
    print("probabilities_logged", probabilities_logged:mean(),probabilities_logged:min(),probabilities_logged:max())
    print("new_probabilities", new_probabilities:mean(),new_probabilities:min(),new_probabilities:max())

    -- Calculate top-1 error, and print information
    print(('Epoch: [%d][%d/%d]\tTime %.3f Reward %.4f RewardsLogged %.4f RewardDiff %.4f  WeightedRewards %.4f WeightedRewardsNew %.4f WeightedRewardsDiff %.4f Top1-%%: %.2f LR %.0e'):format(
        epoch, batch_number, num_batches, timer:time().real,rewards_eva:mean(), rewards:mean(),  diff_rewards,rewards_sum_logged, rewards_sum_new, rewards_sum_new - rewards_sum_logged, top1,
        optimState.learningRate))


    rewards_sum_new_sum = rewards_sum_new_sum  + rewards_sum_new
    rewards_sum_logged_sum = rewards_sum_logged_sum + rewards_sum_logged
    rewards_new_sum = rewards_new_sum + rewards_eva:mean()
    rewards_logged_sum = rewards_logged_sum + rewards:mean()


    return rewards_sum_new,rewards_sum_logged,rewards_eva:mean(), rewards:mean()
end



