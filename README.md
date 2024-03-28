# CFL-Detection

Deploying CFL-Detection Streamlit Application on Heroku.

# Heroku Deployment Guide

## Heroku Login

`heroku login`

## Heroku Container Registry Login

`heroku container:login`

## Build container and push image to Heroku registry

`heroku container:push web -a <heroku-app-name>`

## Release Application

`heroku container:release web -a <heroku-app-name>`

### Scale to Save Money

`heroku ps:scale web=0 -a <heroku-app-name>`

### Scale to Lose Money

`heroku ps:scale web=1 -a <heroku-app-name>`

### Logs

`heroku logs --tail -a <heroku-app-name>`
