----------------------------------------------------------------------
-- This script downloads and loads the MNIST dataset
-- http://yann.lecun.com/exdb/mnist/
----------------------------------------------------------------------


require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'


print '==> downloading dataset'

-- Here we download dataset files.

-- Note: files were converted from their original LUSH format
-- to Torch's internal format.

-- The SVHN dataset contains 3 files:
--    + train: training data
--    + test:  test data

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

if not paths.dirp('mnist.t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

train_file = 'mnist.t7/train_32x32.t7'
test_file = 'mnist.t7/test_32x32.t7'

----------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, it's straightforward

trainData = torch.load(train_file,'ascii')
testData = torch.load(test_file,'ascii')

print('Training Data:')
print(trainData)
print()

print('Test Data:')
print(testData)
print()

----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using itorch.image().
if itorch then
   print('training data:')
   itorch.image(trainData.data[{ {1,256} }])
   print('test data:')
   itorch.image(testData.data[{ {1,256} }])
end


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
paths.dofile('train_bandit.lua')
paths.dofile('test.lua')

epoch = opt.epochNumber

for i=1,opt.nEpochs do
   train()
   test()
   epoch = epoch + 1
end
