#!/bin/sh
set -ex

yt-dlp --remote-components ejs:github -f 243 'https://www.youtube.com/watch?v=FtutLA63Cp8' -o video.webm