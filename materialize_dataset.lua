--
-- Created by IntelliJ IDEA.
-- User: agrotov
-- Date: 4/28/16
-- Time: 2:32 AM
-- To change this template use File | Settings | File Templates.
--

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()
local outputs = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()



function materialize_datase(input_indexes, inputsCPU, labelsCPU, model)
    local parameters, gradParameters = model:getParameters()

    batchNumber = batchNumber or 1

    cutorch.synchronize()
    collectgarbage()

    -- transfer over to GPU
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    local outputs = model:forward(inputs)
    local size_output = outputs:size()
    local actions = sample_action(outputs)

    local p_of_actions= probability_of_actions(outputs, actions)
    local rewards = reward_for_actions(loss_matrix, actions, labels)

    cutorch.synchronize()
    batchNumber = batchNumber + 1

    result = torch.Tensor()

    result = input_indexes

    print(result)
    print(actions)

    result.cat(actions:float(),1)

    print(result)

    return outputs
end

