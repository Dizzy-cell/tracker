python tracker.py --cfg ./configs/actors/duda.yml


git remote add tracker https://github.com/Dizzy-cell/tracker.git
git push -u tracker master


https://labs.play-with-docker.com/

docker pull rootgg/plik
mkdir -p ./data/plik/data
chmod -R 777 ./data/plik/

docker run  -d  \
--name plik \
-p 80:8080  \
-v ./data/plik/data:/home/plik/server/files  \
rootgg/plik

# dataset
1 data 
2 face_landmarker.mask . from mediapipe
3 79999_iter.pth ./faceParse

#output
-input
    -blend
    -mask
    -images

-output
    -checkpoin
    -canonical.obj