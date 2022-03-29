echo "Start run model recognition"
start=`date +%s`


python ocr.py  weights/text-recognition/satrnnet/satrn_vi.py \
--weight weights/text-recognition/satrnnet/epoch_5.pth \
--images TestB1 \
--out_dir results/recognition/satrnnet & 
python ocr.py  weights/text-recognition/RobustScanner/robustscanner_r31.py \
--weight weights/text-recognition/RobustScanner/epoch_5.pth \
--images TestB1 \
--out_dir results/recognition/RobustScanner & 
python ocr.py  weights/text-recognition/NRTR/nrtr_r31_1by8_1by4.py \
--weight weights/text-recognition/NRTR/epoch_5.pth \
--images TestB1 \
--out_dir results/recognition/NRTR
wait
python ensemble_ocr.py 
python post-processing_1.py
python post-processing_2.py

end=`date +%s`

runtime=$((end-start))
echo "Total time $runtime seconds"
echo "End model recognition"