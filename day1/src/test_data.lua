local SimpleReader = require 'SimpleReader'  -- load class

local data_dir = '../data/penn'
local batch_size = 20
local context_size = 2  -- trigram
local min_count = 0

local reader = SimpleReader.create(data_dir, batch_size, context_size, min_count)
reader:load('train') -- loading in training data
print('number of batches: ' .. reader.nbatches)
for i = 1,reader.nbatches do
    local x, y  = reader:next_batch()
end
print('done')
