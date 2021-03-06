--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'cunn'
require'cudnn'
require 'math'


torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

nClasses = opt.nClasses

paths.dofile('util.lua')
paths.dofile('model.lua')
opt.imageSize = model.imageSize or opt.imageSize
opt.imageCrop = model.imageCrop or opt.imageCrop

baseline = opt.baseline

print(opt)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('data.lua')
--paths.dofile('train.lua')
paths.dofile('train_bandit.lua')
paths.dofile('materialize_dataset.lua')
paths.dofile('test.lua')
paths.dofile('load_csv.lua')
paths.dofile('donkey.lua')


function produce_dataset(model, data_path, percentage)
   print("produce_dataset",data_path,opt.epochSize)
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
--   model:training()
   loss_matrix = load_rewards_csv_new("/home/agrotov1/imagenet-multiGPU.torch/loss_matrix.txt")

--   opt.epochSize = 1

--   model:evaluate()
   model:evaluate()

   for i=1,opt.epochSize do
--      local inputs, labels, indexes = trainLoader:sample(opt.batchSize)
--      materialize_datase(indexes, inputs, labels, model, temperature)
        donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels, h1s, w1s, flips, indexes = trainLoader:sample(opt.batchSize, percentage)
            return indexes, inputs, labels, data_path, opt.temperature, h1s, w1s, flips
         end,
         -- the end callback (runs in the main thread)
         materialize_dataset
        )
   end
   donkeys:synchronize()
   save_bandit_dataset(data_path)

end -- of produce_dataset()

local logged_data = nil
local test_logged_data = nil
if opt.produce_dataset ~= 1 and opt.produce_test_dataset ~= 1 then
    logged_data = torch.load(opt.bandit_data)
    test_logged_data = torch.load(opt.bandit_test_data)
end


function compute_variance()

    if opt.variance_reg == 0 then
        return
    end

    mean_so_far= 0
    var_so_far = 0
    count_so_far = 0
    m2_value = 0

    for t = 1,logged_data:size(1),opt.batchSize do
        donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
        --                 logged_data = torch.load(data_path)
              -- create mini batch
            local inputs = torch.Tensor(opt.batchSize,3,opt.cropSize,opt.cropSize)
            local actions = torch.Tensor(opt.batchSize)
            local rewards = torch.Tensor(opt.batchSize)
            local probability_of_actions = torch.Tensor(opt.batchSize)
            local targets = torch.Tensor(opt.batchSize)

            local k = 1
            indexes = torch.Tensor(opt.batchSize,1)

            for i = t,math.min(t+opt.batchSize-1,logged_data:size(1)) do
                local index_of_input = logged_data[i][1]
                local action = logged_data[i][2]
                local reward = logged_data[i][3]
                local probability_of_action = logged_data[i][4]

                local h1 = logged_data[i][5]
                local w1 = logged_data[i][6]
                local flip = logged_data[i][7]


                -- load new sample
                local class = ((index_of_input)%1001)
                local index_of_image = math.floor((index_of_input/1001))
                local input, h1, w1, flip, index_tmp = trainLoader:getByClassAndIndex(class, index_of_image, h1, w1, flip)
                targets[k] = class
                inputs[k] = input
                actions[k] = action
                rewards[k] = reward
                probability_of_actions[k] = probability_of_action

                --            print("class",class,"k",k,"i",i,"math.min(t+opt.batchSize-1,logged_data:size(1))",math.min(t+opt.batchSize-1,logged_data:size(1)))
                k = k + 1
            end

            cutorch.synchronize()
            return inputs,actions,rewards, opt.temperature, probability_of_actions
        end --load_bandit_data,
        ,
         -- the end callback (runs in the main thread)
         compute_variance_batch
        )

        end --for t = 1,logged_data:size(1),opt.batchSize do


        donkeys:synchronize()
        A_w0,b_w0 = get_constants(mean_so_far, variance, count_so_far)
        print("compute_variance",variance,"mean_so_far",mean_so_far)
        print("A_w0",A_w0,"b_w0",b_w0)
--        sys.exit()


end

function train_imagenet_bandit(model, data_path)
   loss_matrix = load_rewards_csv_new("/home/agrotov1/imagenet-multiGPU.torch/loss_matrix.txt")

   epoch = opt.epochNumber

   local last_test_time = sys.clock()


--   rewards_weigted_train_current, rewards_sum_logged_train_current,rewards_new_train_current, rewards_logged_train_current, risk_train_current, risk_logged_train_current = test_imagenet_bandit(model, opt.bandit_train_data)
--   print("EpochTrain 0 rewards_sum",rewards_weigted_train_current,"rewards_sum_logged",rewards_sum_logged_train_current,"rewards",rewards_new_train_current,"rewards_logged",rewards_logged_train_current, "risk", risk_train_current,"risk_logged",risk_logged_train_current)
--
--   rewards_weigted_test_current, rewards_sum_logged_test_current,rewards_new_test_current, rewards_logged_test_current, risk_test_current, risk_logged_test_current = test_imagenet_bandit(model, opt.bandit_test_data)
--   print("EpochTest 0  rewards_sum",rewards_weigted_test_current,"rewards_sum_logged",rewards_sum_logged_test_current,"rewards",rewards_new_test_current,"rewards_logged",rewards_logged_test_current, "risk",risk_test_current,"risk_logged",risk_logged_test_current)

   for i = epoch, opt.nEpochs do
       -- do one epoch
       print('<train_imagenet_bandit> on training set:')
       print("<train_imagenet_bandit> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')


       num_batches = logged_data:size(1)/opt.batchSize
       rewards_sum_new_sum = 0
       rewards_sum_logged_sum = 0
       rewards_new_sum = 0
       rewards_logged_sum = 0
       risk_sum = 0
       risk_logged_sum = 0

       batch_number = 0

       compute_variance()

       for t = 1,logged_data:size(1),opt.batchSize do


          donkeys:addjob(
             -- the job callback (runs in data-worker thread)
             function()
--                 logged_data = torch.load(data_path)
                  -- create mini batch
                local inputs = torch.Tensor(opt.batchSize,3,opt.cropSize,opt.cropSize)
                local actions = torch.Tensor(opt.batchSize)
                local rewards = torch.Tensor(opt.batchSize)
                local probability_of_actions = torch.Tensor(opt.batchSize)
                local targets = torch.Tensor(opt.batchSize)

                local k = 1
                indexes = torch.Tensor(opt.batchSize,1)

                for i = t,math.min(t+opt.batchSize-1,logged_data:size(1)) do
                    local index_of_input = logged_data[i][1]
                    local action = logged_data[i][2]
                    local reward = logged_data[i][3]
                    local probability_of_action = logged_data[i][4]

                    local h1 = logged_data[i][5]
                    local w1 = logged_data[i][6]
                    local flip = logged_data[i][7]


                    -- load new sample
                    local class = ((index_of_input)%1001)
                    local index_of_image = math.floor((index_of_input/1001))
                    local input, h1, w1, flip, index_tmp = trainLoader:getByClassAndIndex(class, index_of_image, h1, w1, flip)
                    targets[k] = class
                    inputs[k] = input
                    actions[k] = action
                    rewards[k] = reward
                    probability_of_actions[k] = probability_of_action

                    --            print("class",class,"k",k,"i",i,"math.min(t+opt.batchSize-1,logged_data:size(1))",math.min(t+opt.batchSize-1,logged_data:size(1)))
                    k = k + 1
                end

                cutorch.synchronize()
                return inputs,actions,rewards,probability_of_actions, targets, opt.temperature, batch_number, opt.baseline
            end --load_bandit_data,
            ,
             -- the end callback (runs in the main thread)
             trainBatch_bandit
            )



       end --for t = 1,logged_data:size(1),opt.batchSize do


       donkeys:synchronize()

       local curr_time = sys.clock()

       local rewards_sum_new_train = rewards_sum_new_sum/batch_number
       local rewards_sum_logged_train = rewards_sum_logged_sum/batch_number
       local rewards_new_train = rewards_new_sum/batch_number
       local rewards_logged_train = rewards_logged_sum/batch_number
       local risk_sum_train = risk_sum/batch_number
       local risk_logged_sum_train = risk_logged_sum/batch_number

       if epoch % 1 == 0 or curr_time - last_test_time > 15 * 60 then
           rewards_sum_new_test,rewards_sum_logged_test,rewards_new_test, rewards_logged_test, risk_sum_test, risk_logged_sum_test = test_imagenet_bandit(model, opt.bandit_test_data)
           last_test_time = sys.clock()
           print("EpochTrain",epoch,"rewards_sum",rewards_sum_new_train,"Delta",rewards_sum_new_train - rewards_sum_logged_train,"rewards",rewards_new_train,"rewards_logged",rewards_logged_train,"risk",risk_sum_train,"risk_logged",risk_logged_sum_train)
           print("EpochTest",epoch,"rewards_sum",rewards_sum_new_test, "Delta", rewards_sum_new_test - rewards_sum_logged_test,"rewards",rewards_new_test,"rewards_logged",rewards_logged_test,"risk",risk_sum_test,"risk_logged",risk_logged_sum_test)

--                   if rewards_sum_new_test - rewards_weigted_test_current < 0 then
--    --                   model:clearState()
--    --                   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
--    --                   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
--    --                   print("no improvement")
--    --                   os.exit()
--                       print("Would terminate epoch",epoch,"rewards_sum_new_test",rewards_sum_new_test,"rewards_sum_logged_test",rewards_sum_logged_test,"rewards_sum_new_test - rewards_sum_logged_test",rewards_sum_new_test - rewards_sum_logged_test,"rewards_new_test",rewards_new_test,"rewards_logged_test",rewards_logged_test)
--                   end

           rewards_weigted_test_current = rewards_sum_new_test
       end --if




--       model:clearState()
--       saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
--       torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
       epoch = epoch + 1
   end --for epoch = epoch or 1, opt.nEpochs do
end -- of train_imagenet_bandit()


function test_imagenet_bandit(model, data_path)

   if not loss_matrix then
       loss_matrix = load_rewards_csv_new("/home/agrotov1/imagenet-multiGPU.torch/loss_matrix.txt")
   end

   epoch = epoch or 1


   model:evaluate()

   print("baseline",baseline)

   rewards_sum_new_sum = 0
   rewards_sum_logged_sum = 0
   rewards_new_sum = 0
   rewards_logged_sum = 0
   batch_number = 0
   risk_sum = 0
   risk_logged_sum = 0
   num_batches = test_logged_data:size(1)/opt.batchSize
   for t = 1,test_logged_data:size(1),opt.batchSize do
        donkeys:addjob(
        -- the job callback (runs in data-worker thread)
        function()
              -- create mini batch
            local inputs = torch.Tensor(opt.batchSize,3,opt.cropSize,opt.cropSize)
            local actions = torch.Tensor(opt.batchSize)
            local rewards = torch.Tensor(opt.batchSize)
            local probability_of_actions = torch.Tensor(opt.batchSize)
            local targets = torch.Tensor(opt.batchSize)

            local k = 1
            indexes = torch.Tensor(opt.batchSize,1)

            for i = t,math.min(t+opt.batchSize-1,test_logged_data:size(1)) do
                local index_of_input = test_logged_data[i][1]
                local action = test_logged_data[i][2]
                local reward = test_logged_data[i][3]
                local probability_of_action = test_logged_data[i][4]

                local h1 = test_logged_data[i][5]
                local w1 = test_logged_data[i][6]
                local flip = test_logged_data[i][7]


                -- load new sample
                local class = ((index_of_input)%1001)
                local index_of_image = math.floor((index_of_input/1001))
                local input, h1, w1, flip, index_tmp = trainLoader:getByClassAndIndex(class, index_of_image, h1, w1, flip)
                targets[k] = class
                inputs[k] = input
                actions[k] = action
                rewards[k] = reward
                probability_of_actions[k] = probability_of_action

                --            print("class",class,"k",k,"i",i,"math.min(t+opt.batchSize-1,logged_data:size(1))",math.min(t+opt.batchSize-1,logged_data:size(1)))
                k = k + 1
            end

            cutorch.synchronize()

            return inputs,actions,rewards,probability_of_actions, targets, opt.temperature
        end --load_bandit_data,
        ,
        -- the end callback (runs in the main thread)
        full_information_full_test
        )
        --      opt.learningRate = 0.01
    end

    donkeys:synchronize()
    cutorch.synchronize()
    local rewards_sum_new = rewards_sum_new_sum/batch_number
    local rewards_sum_logged = rewards_sum_logged_sum/batch_number
    local rewards_new = rewards_new_sum/batch_number
    local rewards_logged = rewards_logged_sum/batch_number



    return rewards_sum_new, rewards_sum_logged, rewards_new, rewards_logged, risk_sum/batch_number, risk_logged_sum/batch_number
end -- of test_imagenet_bandit()










if opt.produce_dataset == 1 then
    produce_dataset(model, opt.bandit_data, 0.9)
--    print_bandit_dataset()
end

if opt.produce_test_dataset == 1 then
    produce_dataset(model, opt.bandit_test_data, -0.1)
--    print_bandit_dataset()
end

if opt.train == 1 then
    train_imagenet_bandit(model,opt.bandit_data)
end


if opt.test == 1 then
    test_imagenet_bandit(model, opt.bandit_data)
end

--print_bandit_dataset()
--

--epoch = opt.epochNumber
--
--for i=1,opt.nEpochs do
--   train()
--   test()
--   epoch = epoch + 1
--end
