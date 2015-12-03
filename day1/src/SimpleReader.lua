--[[
    NMT-Hackathon
    SimpleReader: read in data for neural n-gram language model
    This is a quick implementation, it does not do correct data processing
    such as sentence boundaries, adding <s>, </s>
    WARNING: this is a toy example, do not use it for large dataset
    The SimpleReader is inspried by Kapathy code for char-nn

    author: Ke Tran <m.k.tran@uva.nl>
    date 2/12/2015
--]]
local SimpleReader = {}
SimpleReader.__index = SimpleReader

function SimpleReader.create(data_dir, batch_size, context_size, min_count)
    local self = {}
    setmetatable(self, SimpleReader)  -- this is for creating class in Lua

    -- train, valid, test data
    local train_file = path.join(data_dir, 'train.txt')
    local valid_file = path.join(data_dir, 'valid.txt')
    local test_file = path.join(data_dir, 'test.txt')
    local vocab_file = path.join(data_dir, 'vocab.t7')

    local train_tensor_file = path.join(data_dir, 'train.t7')
    local valid_tensor_file = path.join(data_dir, 'valid.t7')
    local test_tensor_file = path.join(data_dir, 'test.t7')

    -- not that great, but i'm too lazy
    self.train_tensor_file = train_tensor_file
    self.valid_tensor_file = valid_tensor_file
    self.test_tensor_file = test_tensor_file
    self.context_size = context_size
    self.batch_size = batch_size

    local run_prepro = false  -- processing data or not
    if not (path.exists(vocab_file)) then
        print('vocab.t7 does not exist. Run preprocessing!')
        run_prepro = true
    end

    if run_prepro then
        print('one-time setup')
        SimpleReader.text2tensor(train_file, vocab_file, min_count, train_tensor_file)
        SimpleReader.text2tensor(valid_file, vocab_file, min_count, valid_tensor_file)
        SimpleReader.text2tensor(test_file, vocab_file, min_count, test_tensor_file)
    end
    self.word2id = torch.load(vocab_file)
    self.vocab_size = 0
    for _ in pairs(self.word2id) do
        self.vocab_size = self.vocab_size + 1
    end
    collectgarbage()  -- free memory
    return self
end

function SimpleReader:load(mode)
    -- loading data
    local tensor_file
    if mode == 'train' then
        tensor_file = self.train_tensor_file
    elseif mode == 'valid' then
        tensor_file = self.valid_tensor_file
    elseif mode == 'test' then
        tensor_file = self.test_tensor_file
    else
        error('arg: train, valid, or test')
    end
    local data = torch.load(tensor_file)
    -- I'm doing a bit hacky stuff here
    -- ref: https://github.com/torch/torch7/blob/master/doc/tensor.md
    
    local buffer_y = data:sub(self.context_size+1,-1):clone()
    local tot_ngram = buffer_y:nElement()
    self.nbatches = math.floor(tot_ngram/self.batch_size)

    -- truncate the data
    local truncated_len = self.nbatches*self.batch_size
    self.y = buffer_y:sub(1, truncated_len)
    self.x = data:sub(1, data:nElement()-1):unfold(1, self.context_size, 1):narrow(1,1,truncated_len)
    -- set batch offset to 0
    self.curr_batch = 0
    collectgarbage()  -- free up memory again
    return self
end

function SimpleReader:next_batch()
    local x = self.x:narrow(1,self.curr_batch*self.batch_size+1, self.batch_size)
    local y = self.y:narrow(1,self.curr_batch*self.batch_size+1, self.batch_size)
    self.curr_batch = self.curr_batch + 1
    return x,y
end

-- STATIC METHOD
function SimpleReader.text2tensor(in_textfile, out_vocabfile, min_count, out_tensorfile)
    --[[
        A simple transformation of data,
        args:
            in_textfile: input text file that we want to map to tensor
            out_vocabfile: create a vocabulary if it does not exist
            min_count: only put words with frequency > min_count into vocabulary
            out_tensorfile: write out the tensor file. ie. map word -> id
    --]]
    local tot_len = 0  -- total length of data (number of tokens)
    local word_freq = {}  -- word frequency
    for line in io.lines(in_textfile) do
        local ws = stringx.split(line)
        tot_len = tot_len + #ws
        for _,w in pairs(ws) do
            word_freq[w] = (word_freq[w] or 0) + 1
        end
    end

    local word2id  -- map word to id
    if not path.exists(out_vocabfile) then
        print('Creating vocabulary')
        local ordered = {}
        for w,c in pairs(word_freq) do
            if c > min_count then
                ordered[#ordered + 1] = w
            end
        end
        local vocab_size = #ordered  -- vocabulary size
        word2id = {}
        for i,w in pairs(ordered) do word2id[w] = i end
        local special_words = {'<s>', '</s>', '<unk>'}
        for i,w in pairs(special_words) do
            if not word2id[w] then
                vocab_size = vocab_size + 1
                word2id[w] = vocab_size
            end
        end
        print('vocabulary size when created: ' .. vocab_size)
    else
        word2id = torch.load(out_vocabfile)
    end

    local data = torch.IntTensor(tot_len)  -- allocate data
    local curr = 1
    for line in io.lines(in_textfile) do
        local ws = stringx.split(line)
        for _,w in pairs(ws) do
            data[curr] = (word2id[w] or word2id['<unk>'])
            curr = curr + 1
        end
    end
    if not path.exists(out_vocabfile) then
        print('saving ' .. out_vocabfile)
        torch.save(out_vocabfile, word2id)
    end
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, data)
end

return SimpleReader
