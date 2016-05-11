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

function materialize_dataset(input_indexes, inputsCPU, labelsCPU, path, temperature)
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

    print("probabilities")
    print(probabilities[1])
    print("print(torch.sum(probabilities))")
    print(torch.sum(probabilities))

    local size_output = outputs:size()
    local actions = sample_action(outputs,temperature)

    local p_of_actions= probability_of_actions(outputs, actions, temperature)
    print("p_of_actions")
    print(p_of_actions)


    local rewards = reward_for_actions(loss_matrix, actions, labels)

    cutorch.synchronize()
    batchNumber = batchNumber + 1

    result = torch.Tensor()

    result = input_indexes

--    print(result)
--    print(actions:float())

    -- index of input, action , reward, probability
    result = torch.cat(input_indexes,actions:float(),2):cat(rewards:float(), 2):cat(p_of_actions:float(),2)


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