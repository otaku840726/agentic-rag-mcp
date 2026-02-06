#!/usr/bin/env node

/**
 * Post-install script to set up Python dependencies
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const packageRoot = path.resolve(__dirname, '..');

console.log('Setting up agentic-rag-mcp...');

// Check if uv is available
function hasCommand(cmd) {
  try {
    execSync(`${cmd} --version`, { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

async function main() {
  const requirementsPath = path.join(packageRoot, 'requirements.txt');

  if (!fs.existsSync(requirementsPath)) {
    console.log('requirements.txt not found, skipping Python setup.');
    return;
  }

  // Try uv first, then pip
  if (hasCommand('uv')) {
    console.log('Installing Python dependencies with uv...');
    try {
      execSync(`uv pip install -r ${requirementsPath}`, {
        cwd: packageRoot,
        stdio: 'inherit',
      });
      console.log('Python dependencies installed successfully.');
    } catch (e) {
      console.warn('Failed to install with uv, trying pip...');
    }
  }

  if (hasCommand('pip3') || hasCommand('pip')) {
    const pip = hasCommand('pip3') ? 'pip3' : 'pip';
    console.log(`Installing Python dependencies with ${pip}...`);
    try {
      execSync(`${pip} install -r ${requirementsPath}`, {
        cwd: packageRoot,
        stdio: 'inherit',
      });
      console.log('Python dependencies installed successfully.');
    } catch (e) {
      console.error('Failed to install Python dependencies.');
      console.error('Please run manually: pip install -r requirements.txt');
    }
  } else {
    console.warn('Neither uv nor pip found. Please install Python dependencies manually.');
  }
}

main().catch(console.error);
