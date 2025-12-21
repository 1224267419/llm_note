from concurrent.futures import  ThreadPoolExecutor
from fastapi import FastAPI,Request,Response
import uvicorn
import asyncio
import time

# 创建web
app=FastAPI()

# 创建线程池
threadpool=ThreadPoolExecutor(max_workers=200)

# 通过get方法接收参数
@app.get('/ver1')
async def ver1(request:Request):
    # 获取参数
    msg=request.query_params.get('msg')

    # 获取async io event loop
    loop=asyncio.get_event_loop()

    task={
        'msg':msg,
        'event':asyncio.Event(),
        # 'loop':loop,
        'result':None
    }
    # 处理函数,用于处理传入的参数
    def handle_task():
        print("task received",task['msg'])
        # 处理
        task['result']=task['msg'].lower()
        time.sleep(2)#模拟线程阻塞
        def async_callback():
            print("task completed",task['result'],asyncio.get_event_loop())
            # 唤醒
            task['event'].set()
        # 操作event,所以必须放在主线程内
        loop.call_soon_threadsafe(async_callback)
    #   注册方法
    threadpool.submit(handle_task)
    # 等待task唤醒
    await task['event'].wait()

    return Response(task['result'])

@app.get('/ver2')
async def ver2(request:Request):
    # 获取参数
    msg=request.query_params.get('msg')

    # 获取async io event loop
    loop=asyncio.get_event_loop()

    task={
        'msg':msg,

    }
    # 处理函数,用于处理传入的参数
    def handle_task():
        print("task received",task['msg'])
        # 处理
        result=task['msg'].lower()
        time.sleep(0.5)#模拟线程阻塞
        return result

    # 提交并等待结果,这个loop.run_in_executor
    # 封装了上述的大部分操作
    result=await loop.run_in_executor(threadpool,handle_task)
    return Response(result)

if __name__=='__main__':
    uvicorn.run("main:app",host='localhost',port=8000,reload=True)