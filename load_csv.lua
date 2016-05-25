--
-- Created by IntelliJ IDEA.
-- User: agrotov
-- Date: 3/21/16
-- Time: 1:56 PM
-- To change this template use File | Settings | File Templates.
--


function load_rewards_csv(filePath)
    -- Read CSV file

    -- Split string
    function string:split(sep)
      local sep, fields = sep, {}
      local pattern = string.format("([^%s]+)", sep)
      self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
      return fields
    end


    -- Count number of rows and columns in file
    local i = 0
    for line in io.lines(filePath) do
      if i == 0 then
        COLS = #line:split(',')
      end
      i = i + 1
    end
    print("load_rewards_csvload_rewards_csvload_rewards_csv")
    ROWS = i
   --local ROWS = i - 1  -- Minus 1 because of header

    -- Read data from CSV to tensor
    local csvFile = io.open(filePath, 'r')
    print("csvFilecsvFilecsvFilecsvFile")
--    local header = csvFile:read()

    local data = torch.Tensor(ROWS, COLS)
    print("datadatadatadata")
    exit()

    local i = 0
    for line in csvFile:lines('*l') do
      i = i + 1
      print(i)
      local l = line:split(',')
      for key, val in ipairs(l) do
        print(i,key,val)
        data[i][key] = val
      end
    end

    csvFile:close()

    return data
end


