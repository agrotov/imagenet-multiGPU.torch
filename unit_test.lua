--
-- Created by IntelliJ IDEA.
-- User: agrotov
-- Date: 3/24/16
-- Time: 12:46 PM
-- To change this template use File | Settings | File Templates.
--



require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

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
paths.dofile('train.lua')
paths.dofile('test.lua')

epoch = opt.epochNumber



local mytest = torch.TestSuite()

local tester = torch.Tester()

function mytest.testA()
   local a = torch.Tensor{1, 2, 3}
   local b = torch.Tensor{1, 2, 4}
   tester:eq(a, b, "a and b should be equal")
end

-- create fake network
-- create fake data
-- local function sample_action(model_output)
-- reward_for_actions(loss_matrix, actions, labels)
-- probability_of_actions(model_output, actions)

-- compute_weight(rewards, probability_actions_student_model, probability_actions_teacher_model)

-- load_rewards(file_name)

-- compute_target(size, actions, rewards, probability_actions_student_model, probability_actions_teacher_model)

