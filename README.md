# vitmav45-Mayuyu

Building the image:

```
docker build . -t <image-name>
```

Running container:

```
docker run --gpus all -it <image-name>
```

Testing GPU access in container:
```
python project/hello.py
```
