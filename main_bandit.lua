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

   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0

--   opt.epochSize = 1

--   model:evaluate()
   model:evaluate()

   for i=1,opt.epochSize do
--      local inputs, labels, indexes = trainLoader:sample(opt.batchSize)
--      materialize_datase(indexes, inputs, labels, model, temperature)
      print("donkeys:addjob",i)
      local inputs, labels, h1s, w1s, flips, indexes = trainLoader:sample(opt.batchSize, percentage)
      materialize_dataset(indexes, inputs, labels, data_path, opt.temperature, h1s, w1s, flips)
   end
   print("after all")
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
                       1, tm:time().real, loss_epoch, top1_epoch))
   print('\n')

   -- save model
   collectgarbage()
--   print_bandit_dataset()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
end -- of produce_dataset()


local logged_data = torch.load(opt.bandit_data)
local test_logged_data = torch.load(opt.bandit_test_data)

function train_imagenet_bandit(model, data_path)
   loss_matrix = load_rewards_csv_new("/home/agrotov1/imagenet-multiGPU.torch/loss_matrix.txt")

   epoch = opt.epochNumber

   local last_test_time = sys.clock()

--   rewards_weigted_test_current = test_imagenet_bandit(model, opt.bandit_test_data)
--   print("rewards_sum_new_test",rewards_weigted_test_current,"initial")

   for i = epoch, opt.nEpochs do
       -- do one epoch
       print('<train_imagenet_bandit> on training set:')
       print("<train_imagenet_bandit> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')


       num_batches = logged_data:size(1)/opt.batchSize
       rewards_sum_new_sum = 0
       rewards_sum_logged_sum = 0
       rewards_new_sum = 0
       rewards_logged_sum = 0

       batch_number = 0

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

       if epoch % 1 == 0 or curr_time - last_test_time > 15 * 60 then
           rewards_sum_new_test,rewards_sum_logged_test,rewards_new_test, rewards_logged_test = test_imagenet_bandit(model, opt.bandit_test_data)
           last_test_time = sys.clock()

           print("epoch",epoch,"rewards_sum_new_test",rewards_sum_new_test,"rewards_sum_logged_test",rewards_sum_logged_test,"rewards_sum_new_test - rewards_sum_logged_test",rewards_sum_new_test - rewards_sum_logged_test,"rewards_new_test",rewards_new_test,"rewards_logged_test",rewards_logged_test,"batch_number",batch_number)

                   if rewards_sum_new_test - rewards_weigted_test_current < 0 then
    --                   model:clearState()
    --                   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
    --                   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
    --                   print("no improvement")
    --                   os.exit()
                       print("Would terminate epoch",epoch,"rewards_sum_new_test",rewards_sum_new_test,"rewards_sum_logged_test",rewards_sum_logged_test,"rewards_sum_new_test - rewards_sum_logged_test",rewards_sum_new_test - rewards_sum_logged_test,"rewards_new_test",rewards_new_test,"rewards_logged_test",rewards_logged_test)
                   end

           rewards_weigted_test_current = rewards_sum_new_test
       end --if


       local rewards_sum_new_train = rewards_sum_new_sum/batch_number
       local rewards_sum_logged_train = rewards_sum_logged_sum/batch_number
       local rewards_new_train = rewards_new_sum/batch_number
       local rewards_logged_train = rewards_logged_sum/batch_number

       print("epoch",epoch,"rewards_sum_new_train",rewards_sum_new_train,"rewards_sum_new_train - rewards_sum_logged_train",rewards_sum_new_train - rewards_sum_logged_train,"rewards_new_train",rewards_new_train,"rewards_logged_train",rewards_logged_train,"batch_number",batch_number)


       model:clearState()
       saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
       torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
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


    return rewards_sum_new, rewards_sum_logged, rewards_new, rewards_logged
end -- of test_imagenet_bandit()










if opt.produce_dataset == 1 then
    produce_dataset(model, data_path, 0.9)
--    print_bandit_dataset()
end

if opt.produce_test_dataset == 1 then
    produce_dataset(model, data_path, -0.1)
--    print_bandit_dataset()
end

if opt.train == 1 then
    train_imagenet_bandit(model,data_path)
end


if opt.test == 1 then
    test_imagenet_bandit(model, data_path)
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
