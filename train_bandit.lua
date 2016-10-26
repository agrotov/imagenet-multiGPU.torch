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
--    probability_actions_teacher_model:clamp(0.01, torch.max(probability_actions_teacher_model))
    local propencity = torch.cdiv(probability_actions_student_model,probability_actions_teacher_model)
    return -torch.cmul(rewards_arg,propencity)
--    return rewards_arg
end

function load_rewards(file_name)
    return csvigo.load({path = file_name, mode = "large"})
end


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


mean_so_far= 0
number_of_data_processed = 0
m2_value = 0
A_w0 = 0
b_w0 = 0


function compute_variance_batch(inputsCPU, actions_cpu, rewards_cpu, temperature)
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


    for i=1,opt.batchSize do
        weighted_reward_value = weighted_reward[i][1]

        number_of_data_processed  = number_of_data_processed  + 1.0

        delta = weighted_reward_value - mean_so_far

        mean_so_far = mean_so_far + delta/number_of_data_processed

        m2_value = m2_value + delta*(weighted_reward_value - mean_so_far)
    end

end



function get_constants(mean_weighted_rewards, variance, num_examples)
    sqrt_variance = torch.sqrt(variance)
    A_w0 = - mean_weighted_rewards / ((num_examples - 1) * sqrt_variance)
    b_w0 = 1/(2 * (num_examples - 1) * sqrt_variance)

    return A_w0,b_w0
end

function get_variance_gradient(weighted_rewards, grad_of_weighted_rewards)
    var_grad = A_w0*grad_of_weighted_rewards + 2 * b_w0 * torch.cmul(weighted_rewards , grad_of_weighted_rewards)
    return var_grad
end


function compute_target(outputs, size, actions, rewards_arg, probability_actions_student_model, probability_actions_teacher_model, baseline)
--    target = torch.Tensor(size):fill(0)
    probability_actions_teacher_model_clamped = torch.clamp(probability_actions_teacher_model,0.01, torch.max(probability_actions_teacher_model))

--    weight = compute_weight(rewards_arg-opt.baseline, probability_actions_student_model, probability_actions_teacher_model_clamped)

    local propencity = torch.cdiv(probability_actions_student_model,probability_actions_teacher_model_clamped)
    target = -torch.cmul(rewards_arg-opt.baseline,propencity)
--    target:scatter(2,actions:long(),weight:float())

    gradient_of_risk = -torch.cdiv(rewards_arg-opt.baseline,probability_actions_teacher_model_clamped)
--    gradient_of_risk_scattered =torch.Tensor(size):fill(0)
--    gradient_of_risk_scattered:scatter(2,actions:long(),gradient_of_risk:float())

    print("target",target:mean(),target:min(),target:max())
    print("gradient_of_risk",gradient_of_risk:mean(),gradient_of_risk:min(),gradient_of_risk:max())

--    expected_reward = torch.cmul(probability_actions_student_model,rewards_arg-opt.baseline)
--    expected_reward_scattered = torch.Tensor(size):fill(0)
--    expected_reward_scattered:scatter(2,actions:long(),expected_reward:float())

    variance_grad = get_variance_gradient(target, gradient_of_risk)

    variace_regularised_target = target + opt.variance_reg * variance_grad

    print("variace_regularised_target",variace_regularised_target:mean(),variace_regularised_target:min(),variace_regularised_target:max())

    log_probability_of_actions_val = log_probability_of_actions(outputs, actions)
--    log_probability_of_actions_val_scattered = torch.Tensor(size):fill(0)
--    log_probability_of_actions_val_scattered:scatter(2,actions:long(),log_probability_of_actions_val:float())

    print("log_probability_of_actions_val",log_probability_of_actions_val:mean(),log_probability_of_actions_val:min(),log_probability_of_actions_val:max())

    new_target = -torch.cdiv(variace_regularised_target, log_probability_of_actions_val)

    new_target_scattered = torch.Tensor(size):fill(0)
    new_target_scattered:scatter(2,actions:long(),new_target:float())

    print("new_target",new_target_scattered:mean(),new_target_scattered:min(),new_target_scattered:max())

    return new_target

--    target:scatter(2,actions:long(),weight:float())

--    return target + opt.variance_reg * variance_grad
--    return target
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
    print("trainBatch_bandit A_w0",A_w0)
    print("trainBatch_bandit b_w0",b_w0)


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
        print("gpu_target",gpu_target:mean(),gpu_target:min(),gpu_target:max())
        print("gradParameters_fresh", gradParameters:mean(),gradParameters:min(),gradParameters:max())

--        nan_mask = gradParameters:ne(gradParameters)
--        non_nan_mask = gradParameters:eq(gradParameters)
--        print("sum nan ",torch.sum(nan_mask),torch.sum(non_nan_mask))
        gradParameters = gradParameters/torch.abs(gradParameters:max())
        print("gradParameters", gradParameters:mean(),gradParameters:min(),gradParameters:max())
        print("parameters", parameters:mean(),parameters:min(),parameters:max())
--        gradParameters:clamp(-5, 5)

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

    print("new_probabilities",torch.cat(probabilities_logged,torch.cat(new_probabilities,rewards,2),2))

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



