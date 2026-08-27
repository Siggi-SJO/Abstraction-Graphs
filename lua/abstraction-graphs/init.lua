local M = {}

local function is_html_file(name)
  return name:match('%.html$') ~= nil
end

local function find_graph_file_in(graph_dir, keyword, is_ext)
  if vim.fn.isdirectory(graph_dir) ~= 1 then
    return nil
  end
  local entries = vim.fn.readdir(graph_dir)
  table.sort(entries)
  for _, name in ipairs(entries) do
    if is_ext(name) and name:lower():find(keyword, 1, true) then
      return graph_dir .. '/' .. name
    end
  end
  return nil
end

-- Search `start_dir`, then each parent in turn, for a `dependency_graph` folder
-- containing a file matching `keyword`. Used to locate call_graph.html for browser open.
local function find_graph_upward(start_dir, keyword, is_ext)
  local dir = vim.fn.fnamemodify(start_dir, ':p'):gsub('/$', '')
  while dir ~= '' and dir ~= '/' do
    local found = find_graph_file_in(dir .. '/dependency_graph', keyword, is_ext)
    if found then
      return found
    end
    local parent = vim.fn.fnamemodify(dir, ':h')
    if parent == dir then
      break
    end
    dir = parent
  end
  return nil
end

-- The nearest directory (starting here, then upward) that is itself a Python module
-- (has its own __init__.py).
local function find_module_dir_upward(start_dir)
  local dir = vim.fn.fnamemodify(start_dir, ':p'):gsub('/$', '')
  while dir ~= '' and dir ~= '/' do
    if vim.fn.filereadable(dir .. '/__init__.py') == 1 then
      return dir
    end
    local parent = vim.fn.fnamemodify(dir, ':h')
    if parent == dir then
      break
    end
    dir = parent
  end
  return nil
end

-- The import root for a module: climb past every ancestor that is itself a package
-- (has __init__.py) -- the first ancestor that doesn't is the directory Python treats
-- as the import root.
local function find_repo_root_upward(module_dir)
  local dir = module_dir
  while true do
    local parent = vim.fn.fnamemodify(dir, ':h')
    if parent == dir then
      return dir
    end
    if vim.fn.filereadable(parent .. '/__init__.py') == 1 then
      dir = parent
    else
      return parent
    end
  end
end

-- The directory to generate from:
--  - in nvim-tree: the hovered node's directory
--  - elsewhere: the current buffer's directory
local function resolve_base_dir()
  if vim.bo.filetype == 'NvimTree' then
    local ok, api = pcall(require, 'nvim-tree.api')
    if not ok then
      vim.notify('nvim-tree.lua not found', vim.log.levels.WARN)
      return nil
    end
    local node = api.tree.get_node_under_cursor()
    if not node then
      vim.notify('No nvim-tree node under cursor', vim.log.levels.WARN)
      return nil
    end
    if node.type == 'directory' then
      return node.absolute_path
    end
    return vim.fn.fnamemodify(node.absolute_path, ':h')
  end

  local bufname = vim.api.nvim_buf_get_name(0)
  if bufname == '' then
    vim.notify('No file in the current buffer', vim.log.levels.WARN)
    return nil
  end
  return vim.fn.fnamemodify(bufname, ':h')
end

function M.setup(opts)
  opts = opts or {}
  local plugin_root = vim.fn.fnamemodify(debug.getinfo(1, 'S').source:sub(2), ':p:h:h:h')
  local python = opts.python or (plugin_root .. '/.venv/bin/python')
  local server_script = opts.server_script or (plugin_root .. '/python/server.py')

  local function open_interactive_call_graph()
    local base_dir = resolve_base_dir()
    if not base_dir then
      return
    end
    local target_path = find_graph_upward(base_dir, 'call', is_html_file)
    if not target_path then
      vim.notify(
        'No interactive call graph (call_graph.html) found searching upward from ' .. base_dir,
        vim.log.levels.WARN
      )
      return
    end
    vim.ui.open(target_path)
  end

  local function generate_graphs(force)
    local base_dir = resolve_base_dir()
    if not base_dir then
      return
    end
    local module_dir = find_module_dir_upward(base_dir)
    if not module_dir then
      vim.notify('No Python module (__init__.py) found searching upward from ' .. base_dir, vim.log.levels.WARN)
      return
    end
    local repo_root = find_repo_root_upward(module_dir)
    local target = module_dir:sub(#repo_root + 2)

    local cmd = { python, server_script, target, '--repo-root', repo_root }
    if force then
      table.insert(cmd, '--force')
    end

    vim.notify(
      'Generating graphs for ' .. target .. (force and ' (forced)' or '') .. '...',
      vim.log.levels.INFO
    )
    local output = {}
    vim.fn.jobstart(cmd, {
      stdout_buffered = true,
      stderr_buffered = true,
      on_stdout = function(_, data)
        vim.list_extend(output, data)
      end,
      on_stderr = function(_, data)
        vim.list_extend(output, data)
      end,
      on_exit = function(_, code)
        local text = table.concat(output, '\n')
        if code == 0 then
          vim.notify('Generated graphs for ' .. target .. '\n' .. text, vim.log.levels.INFO)
        else
          vim.notify(
            'Graph generation failed for ' .. target .. ' (exit ' .. code .. ')\n' .. text,
            vim.log.levels.ERROR
          )
        end
      end,
    })
  end

  vim.keymap.set('n', '<leader>mi', open_interactive_call_graph, { desc = 'Abstraction graphs: open interactive call graph' })
  vim.keymap.set('n', '<leader>mg', function()
    generate_graphs(false)
  end, { desc = 'Abstraction graphs: generate graphs for module' })
  vim.keymap.set('n', '<leader>mF', function()
    generate_graphs(true)
  end, { desc = 'Abstraction graphs: force-regenerate graphs for module' })
end

return M
