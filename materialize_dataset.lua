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

    batchNumber = batchNumber or 1

    cutorch.synchronize()
    collectgarbage()

    local num_actions = 1000

    local num_inputs = input_indexes:size()[1]

    local full_size = input_indexes:size()
    full_size[1] = full_size[1] * num_actions

    local input_indexes_full = torch.Tensor(full_size)


    -- transfer over to GPU

    local actions_taken = torch.LongTensor(num_actions)

    -- fill in with 1,2,3
    local incrementor = 0
    actions_taken:apply(function() incrementor  = incrementor  + 1; return incrementor  end)

    local p_of_action = 1.0/num_actions
    local p_of_actions= torch.Tensor(num_actions):fill(p_of_action)

    print("materialize_full_dataset",num_inputs)

    for input_index=1,num_inputs do
--        print("input_index",input_index)
        local image_index = input_indexes[input_index][1]
        local input_indexes_full = torch.Tensor(num_actions):fill(image_index)

        local label_of_input = labelsCPU[input_index]
        local label_of_input_tensor = torch.Tensor(num_actions):fill(label_of_input)


        local rewards = reward_for_actions(loss_matrix, actions_taken, label_of_input_tensor)

--        print("rewards",rewards[label_of_input])

--        print("loss_matrix")
--        print(loss_matrix)

        local h1s_full= torch.Tensor(num_actions):fill(h1s[input_index][1])
        local w1s_full= torch.Tensor(num_actions):fill(w1s[input_index][1])
        local flips1s_full= torch.Tensor(num_actions):fill(flips[input_index][1])

        result_for_input = torch.cat(input_indexes_full,actions_taken:float(),2):cat(rewards:float(), 2):cat(p_of_actions:float(),2):cat(h1s_full:float(),2):cat(w1s_full:float(),2):cat(flips1s_full:float(),2)

--        print(result_for_input[{{1,4}}])
        print("actions_taken",actions_taken)
--        exit()
        if bandit_dataset ~= nil then
            bandit_dataset = bandit_dataset:cat(result_for_input,1)
        else
            bandit_dataset = result_for_input:clone()
        end
    end
    exit()
end


function save_bandit_dataset(filename)
    torch.save(filename,bandit_dataset)
end

function print_bandit_dataset()
    print("bandit_dataset")
    print(bandit_dataset)
end