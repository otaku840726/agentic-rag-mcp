#!/usr/bin/env node

/**
 * Agentic RAG MCP Server - Node.js wrapper
 * This script spawns the Python MCP server using uvx or python directly
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Find the package root
const packageRoot = path.resolve(__dirname, '..');

// Check for Python virtual environment
const venvPython = path.join(packageRoot, 'venv', 'bin', 'python');
const venvPythonWin = path.join(packageRoot, 'venv', 'Scripts', 'python.exe');

async function findPython() {
  // Priority: uvx > venv > system python
  const candidates = [
    { cmd: 'uvx', args: ['--from', 'agentic-rag-mcp', 'agentic-rag-mcp'], type: 'uvx' },
    { cmd: venvPython, args: ['-m', 'agentic_rag_mcp'], type: 'venv' },
    { cmd: venvPythonWin, args: ['-m', 'agentic_rag_mcp'], type: 'venv' },
    { cmd: 'python3', args: ['-m', 'agentic_rag_mcp'], type: 'system' },
    { cmd: 'python', args: ['-m', 'agentic_rag_mcp'], type: 'system' },
  ];

  for (const candidate of candidates) {
    try {
      if (candidate.type === 'venv' && !fs.existsSync(candidate.cmd)) {
        continue;
      }
      return candidate;
    } catch (e) {
      continue;
    }
  }

  throw new Error('Python not found. Please install Python 3.10+ or uv.');
}

async function main() {
  try {
    const python = await findPython();

    // Spawn the Python process
    const proc = spawn(python.cmd, python.args, {
      cwd: packageRoot,
      stdio: 'inherit',
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
      },
    });

    proc.on('error', (err) => {
      console.error(`Failed to start MCP server: ${err.message}`);
      process.exit(1);
    });

    proc.on('exit', (code) => {
      process.exit(code || 0);
    });

    // Forward signals
    process.on('SIGINT', () => proc.kill('SIGINT'));
    process.on('SIGTERM', () => proc.kill('SIGTERM'));

  } catch (err) {
    console.error(`Error: ${err.message}`);
    console.error('\nTo install, run:');
    console.error('  pip install agentic-rag-mcp');
    console.error('  # or with uv:');
    console.error('  uv pip install agentic-rag-mcp');
    process.exit(1);
  }
}

main();
