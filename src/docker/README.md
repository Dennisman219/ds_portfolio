# How To

1. Run `docker build -t prediction .`
2. Run `docker run -p 5050:5050 prediction`
3. Should now be running

Now you should be able to do post requests to the API.

Example:

`curl -L -X POST -H "Content-Type: application/json" -d '{ "posts_list": "mdma xtc weed\nselling cook book and politics book" }'`