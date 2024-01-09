- 默认设置:zero2 offload:false,EPOCH=2,GPU*1,DEV_BATCH_SIZE=1,MAX_SOURCE_LEN=8000,GRAD_ACCUMULARION_STEPS=4情况下,6-7s一个step,
-- DEV_BATCH_SIZE=2时则~50s一个step
-- GRAD_ACCUMULARION_STEPS=16 则25s一个step
-- GPU*4     10s一个step,总耗时60h
-- zero 0    GPU*4 8s一个step,大概43个小时完成
-- zero 0    GPU*4 DEV_BATCH_SIZE=2 50s一个step,大概134小时完成 pass


