#!/bin/bash
# ssh -i ~/.ssh/lambda-key.pem ubuntu@150.136.130.35

#git clone https://github.com/your-username/Unrolled-DOT.git
#cd Unrolled-DOT
#
## Optional: create folders if not present (safety)
#mkdir -p Unrolled-DOT-files/datapath
#mkdir -p Unrolled-DOT-files/libpath
#mkdir -p Unrolled-DOT-files/results
# === CONFIGURE THESE ===
PEM_FILE=~/.ssh/lambda-key.pem                          # Your private key
INSTANCE_IP=150.136.44.60                  # Lambda Labs instance IP
REMOTE_USER=ubuntu
LOCAL_PROJECT_PATH=/Users/prajakta/mscs_fall2024/sem1/Unrolled-DOT
REMOTE_PROJECT_ROOT=~/Unrolled-DOT                      # Base folder on Lambda
REMOTE_PROJECT_PATH=$REMOTE_PROJECT_ROOT/unrolled_DOT_code  # Folder where Makefile lives
MAKE_TARGET=all                                         # Target to run (default: all)

# === 🔒 Step 1: Ensure correct permissions on key ===
chmod 400 $PEM_FILE

# === 📤 Step 2: Sync project to remote ===
echo "📤 Syncing local project to Lambda instance..."
rsync -avz -e "ssh -i $PEM_FILE" $LOCAL_PROJECT_PATH/ $REMOTE_USER@$INSTANCE_IP:$REMOTE_PROJECT_ROOT

# === 🚀 Step 3: SSH and run Makefile ===
echo "🚀 Connecting to Lambda and running make..."

ssh -i $PEM_FILE $REMOTE_USER@$INSTANCE_IP << EOF
echo "✅ Connected to Lambda"

#cd $REMOTE_PROJECT_PATH
#
## Optional: activate conda env
## source ~/miniconda3/etc/profile.d/conda.sh
## conda activate unrolled-dot-env
#
#echo "🛠️  Running make $MAKE_TARGET..."
#make $MAKE_TARGET
EOF