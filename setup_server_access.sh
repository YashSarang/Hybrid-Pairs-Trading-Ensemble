#!/bin/bash
# Setup SSH access to Kalpana cluster
# Run this script manually to set up passwordless SSH

echo "Setting up SSH access to kalpana.minds.iitb.ac.in"
echo ""

# Copy SSH key to server
echo "Copying SSH key to server (you'll need to enter password: yash.sarang)"
ssh-copy-id yash.sarang@kalpana.minds.iitb.ac.in

echo ""
echo "Testing connection..."
ssh yash.sarang@kalpana.minds.iitb.ac.in 'echo "✓ Connected to $(hostname)"; pwd'

echo ""
echo "✓ Setup complete! You can now use 'ssh kalpana' for quick access"
