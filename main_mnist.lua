require 'torch'
require 'cutorch'
-- require 'nn'
require 'nnx'
require 'cunn'
require'cudnn'
require 'optim'
require 'image'
-- require 'dataset-mnist'
require 'paths'
paths.dofile('dataset-mnist.lua')
require 'pl'
--torch.setdefaulttensortype('torch.CudaTensor')
cutorch.setDevice(1) -- by default, use GPU 1

----------------------------------------------------------------------
-- parse command-line options
--
opt = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 10)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
]]


opt.epochSize = 1

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
   error('LBFGS should not be used with small mini-batches; 1000 is recommended')
end

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32 }

--opt.model = 'linear'

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()

   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network 
      ------------------------------------------------------------
      -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer MLP:
      model:add(nn.Reshape(64*2*2))
      model:add(nn.Linear(64*2*2, 200))
      model:add(nn.Tanh())
      model:add(nn.Linear(200, #classes))
      ------------------------------------------------------------

   elseif opt.model == 'mlp' then
      ------------------------------------------------------------
      -- regular 2-layer MLP
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024, 2048))
      model:add(nn.Tanh())
      model:add(nn.Linear(2048,#classes))
      ------------------------------------------------------------

   elseif opt.model == 'linear' then
      ------------------------------------------------------------
      -- simple linear model: logistic regression
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024,#classes))
      ------------------------------------------------------------

   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)

end




-- retrieve parameters and gradients
--parameters,gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')


----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
model:cuda()
cudnn.convert(model, cudnn)
criterion = nn.ClassNLLCriterion()
criterion:cuda()


--paths.dofile('train.lua')
paths.dofile('train_bandit.lua')
paths.dofile('materialize_dataset.lua')
loss_matrix = load_rewards_mnist()


print(model)
----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 2000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- training function

function train_mnist(dataset)
      epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      indexes = torch.Tensor(opt.batchSize,1)

      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
--         print(k)
         indexes[k][1] = i

         k = k + 1
      end


--      opt.learningRate = 0.01

      cutorch.synchronize()
      optimState = sgdState or {
         learningRate = opt.learningRate,
         momentum = opt.momentum,
         learningRateDecay = 5e-7
      }


--      outputs = trainBatch_full(inputs,targets, optimState)
      outputs = materialize_datase(indexes,inputs,targets, model)

      -- disp progress
      xlua.progress(t, dataset:size())

      -- update confusion
      for i = 1,opt.batchSize do
         confusion:add(outputs[i], targets[i])
      end
   end

   -- time taken

   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'mnist.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

function train_mnist_bandit(dataset,logged_data)
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   local batch_size = 10


   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,logged_data:size(1),opt.batchSize do

      print("t")
      print(t)
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      indexes = torch.Tensor(opt.batchSize,1)

      for i = t,math.min(t+opt.batchSize-1,logged_data:size(1)) do
            print(i)
            print(logged_data[i])
            index_of_input = logged_data[i][1]
            print(index_of_input)
         -- load new sample
--         local sample = dataset[i]
--         local input = sample[1]:clone()
--         local _,target = sample[2]:clone():max(1)
--         target = target:squeeze()
--         inputs[k] = input
--         targets[k] = target
----         print(k)
--         indexes[k][1] = i
--
--         k = k + 1
      end


--      opt.learningRate = 0.01

--      cutorch.synchronize()
--      optimState = sgdState or {
--         learningRate = opt.learningRate,
--         momentum = opt.momentum,
--         learningRateDecay = 5e-7
--      }
--
--
----      outputs = trainBatch_full(inputs,targets, optimState)
--      outputs = materialize_datase(indexes,inputs,targets, model)
--
--      -- disp progress
--      xlua.progress(t, dataset:size())
--
--      -- update confusion
--      for i = 1,opt.batchSize do
--         confusion:add(outputs[i], targets[i])
--      end
   end
--
--   -- time taken
--
--   time = sys.clock() - time
--   time = time / dataset:size()
--   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
--
--   -- print confusion matrix
--   print(confusion)
--   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
--   confusion:zero()
--
--   -- save/log current net
--   local filename = paths.concat(opt.save, 'mnist.net')
--   os.execute('mkdir -p ' .. sys.dirname(filename))
--   if paths.filep(filename) then
--      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
--   end
--   print('<trainer> saving network to '..filename)
--   torch.save(filename, model)
--
--   -- next epoch
--   epoch = epoch + 1
end




-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- test samples
      local preds = model:forward(inputs:cuda())

      -- confusion:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()


end

----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   logged_data = torch.load("/var/scratch/agrotov/bandit_mnist/mnist_bandit_dataset")
   train_mnist_bandit(dataset,logged_data)
   if (epoch > 1) then
--      save_bandit_dataset("/var/scratch/agrotov/bandit_mnist/mnist_bandit_dataset")
--      print(bandit_dataset)
      exit()
   end

--   test(testData)

   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end