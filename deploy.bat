@echo off

REM Hardcoded Heroku app name
set APP_NAME=cfl-detection

echo Pushing Docker container to Heroku...
heroku container:push web -a %APP_NAME%

echo Releasing Docker container on Heroku...
heroku container:release web -a %APP_NAME%

echo Deployment complete!
