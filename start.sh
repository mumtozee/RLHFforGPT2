export PYTHONPATH=$PYTHONPATH:$(pwd)/proto_lib
echo $PYTHONPATH
(CUDA_VISIBLE_DEVICES=6 python main.py hh:dpo --fastapi_port 8898) & 
pid=$!
sleep 40s && 
python request_example.py --library=asyncio
kill -SIGKILL $pid