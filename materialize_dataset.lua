--
-- Created by IntelliJ IDEA.
-- User: agrotov
-- Date: 4/28/16
-- Time: 2:32 AM
-- To change this template use File | Settings | File Templates.
--
require 'cutorch'
-- require 'nn'
require 'nnx'
require 'cunn'
require'cudnn'


local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()
local outputs = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()



--bandit_dataset = nil

function materialize_dataset(input_indexes, inputsCPU, labelsCPU, path, temperature, h1s, w1s, flips)
    temperature = temperature or 0.5
    local parameters, gradParameters = model:getParameters()

    batchNumber = batchNumber or 1

    cutorch.synchronize()
    collectgarbage()

    -- transfer over to GPU
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    local outputs = model:forward(inputs)

    probabilities = probabilities_from_output(outputs, temperature)

--    print("probabilities")
--    print(probabilities[1])
--    print("print(torch.sum(probabilities[1]))")
--    print(torch.sum(probabilities[1]))

    local size_output = outputs:size()
    local actions = sample_action(outputs,temperature)

    local p_of_actions= probability_of_actions(outputs, actions, temperature)
--    print("p_of_actions")
--    print(p_of_actions)

--    print(torch.cat(a,a,2):cat(p_of_actions,2))


    local rewards = reward_for_actions(loss_matrix, actions, labels)

    cutorch.synchronize()
    batchNumber = batchNumber + 1

    result = torch.Tensor()

    result = input_indexes

--    print(result)
--    print(actions:float())

    -- index of input, action , reward, probability
    result = torch.cat(input_indexes,actions:float(),2):cat(rewards:float(), 2):cat(p_of_actions:float(),2):cat(h1s:float(),2):cat(w1s:float(),2):cat(flips:float(),2)


    if bandit_dataset ~= nil then
        bandit_dataset = bandit_dataset:cat(result,1)
    else
        bandit_dataset = result:clone()
    end
    save_bandit_dataset(path)
    return outputs
end



function materialize_full_dataset(input_indexes, inputsCPU, labelsCPU, path, temperature, h1s, w1s, flips)
    temperature = temperature or 0.5

    batchNumber = batchNumber or 1

    cutorch.synchronize()
    collectgarbage()

    num_actions = 1000

    print("inputsCPU:size()")
    print(inputsCPU:size())
    exit()

    num_inputs = input_indexes:size()[1]

    full_size = input_indexes:size()
    full_size[1] = full_size[1] * num_actions

    input_indexes_full = torch.Tensor(full_size)


    -- transfer over to GPU
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    local outputs = model:forward(inputs)
    probabilities = probabilities_from_output(outputs, temperature)

    print("probabilities:size()")
    print(probabilities:size())

    for input_index=1,num_actions do
        exit()
    end






--    print("probabilities")
--    print(probabilities[1])
--    print("print(torch.sum(probabilities[1]))")
--    print(torch.sum(probabilities[1]))

    local size_output = outputs:size()
    local actions = sample_action(outputs,temperature)

    local p_of_actions= probability_of_actions(outputs, actions, temperature)
--    print("p_of_actions")
--    print(p_of_actions)

--    print(torch.cat(a,a,2):cat(p_of_actions,2))


    local rewards = reward_for_actions(loss_matrix, actions, labels)

    cutorch.synchronize()
    batchNumber = batchNumber + 1

    result = torch.Tensor()

    result = input_indexes

--    print(result)
--    print(actions:float())

    -- index of input, action , reward, probability
    result = torch.cat(input_indexes,actions:float(),2):cat(rewards:float(), 2):cat(p_of_actions:float(),2):cat(h1s:float(),2):cat(w1s:float(),2):cat(flips:float(),2)


    if bandit_dataset ~= nil then
        bandit_dataset = bandit_dataset:cat(result,1)
    else
        bandit_dataset = result:clone()
    end
    save_bandit_dataset(path)
    return outputs
end


function save_bandit_dataset(filename)
    torch.save(filename,bandit_dataset)
end

function print_bandit_dataset()
    print("bandit_dataset")
    print(bandit_dataset)
end