#!/bin/bash

SOURCE="./"
TARGET="ptolsma@student-linux.tudelft.nl:/home/nfs/ptolsma/storage/mvster"
EXCLUDE_FOLDERS=(logs/ outputs/ .venv/ .git/ .idea/ .vscode/ __pycache__/ img)
# Construct the exclude options from the excluded folders
EXCLUDE_OPTIONS=""
for FOLDER in "${EXCLUDE_FOLDERS[@]}"
do
    EXCLUDE_OPTIONS="$EXCLUDE_OPTIONS --exclude '$FOLDER'"
done

#echo $EXCLUDE_OPTIONS
echo rsync -rvz $EXCLUDE_OPTIONS "$SOURCE" "$TARGET"
