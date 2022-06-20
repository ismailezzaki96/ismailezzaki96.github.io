#!/bin/bash

echo "Total arguments : $#"
echo "1st Argument = $1"

youtube-dl -f 'bestvideo[height<=240]+bestaudio/best[height<=240]' -o "/home/k/Desktop/python scripts/youtube/%(title)s.%(ext)s"  --all-subs  --sub-format "srt" $1 
