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

function produce_dataset(model)
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
--   model:training()

   loss_matrix = load_rewards_csv("/home/agrotov/imagenet-multiGPU.torch/loss_matrix.txt")

   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0

--   opt.epochSize = 1

   temperature = 1

   for i=1,opt.epochSize do
--      local inputs, labels, indexes = trainLoader:sample(opt.batchSize)
--      materialize_datase(indexes, inputs, labels, model, temperature)
      print("donkeys:addjob")
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels, indexes = trainLoader:sample(opt.batchSize)
            print("donkeys:addjob sample")
            return indexes, inputs, labels
         end,
         -- the end callback (runs in the main thread)
         materialize_dataset
      )

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

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
end -- of produce_dataset()




function train_imagenet_bandit(model)

   paths.dofile('donkey.lua')

   logged_data = torch.load("/var/scratch/agrotov/bandit_imagenet/logged_dataset_small")

   epoch = epoch or 1
   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,logged_data:size(1),opt.batchSize do

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

         -- load new sample
--         print("class")
--         print(index_of_input)
--         print(((index_of_input-1)%1000)+1)
--         print(math.floor((index_of_input-1)/1000))
         local input, index_tmp = trainLoader:getByClassAndIndex(((index_of_input-1)%1000)+1, math.floor((index_of_input-1)/1000))
         inputs[k] = input
         actions[k] = action
         rewards[k] = reward
         probability_of_actions[k] = probability_of_action
         k = k + 1
      end

      opt.learningRate = 0.01

      cutorch.synchronize()
      optimState = sgdState or {
         learningRate = opt.learningRate,
         momentum = opt.momentum,
         learningRateDecay = 5e-7
      }
      print("trainBatch_bandit")
      print(t)

      outputs = trainBatch_bandit(inputs,actions,rewards,probability_of_actions, optimState)

   end

   -- time taken

   time = sys.clock() - time
   time = time / logged_data:size(1)
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

end -- of train_imagenet_bandit()




--produce_dataset(model)

train_imagenet_bandit(model)


--epoch = opt.epochNumber
--
--for i=1,opt.nEpochs do
--   train()
--   test()
--   epoch = epoch + 1
--end
