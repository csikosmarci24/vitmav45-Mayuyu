# Team information

Team name: Mayuyu

Members:

| Name | NEPTUN |
| ---- | ------ |
| Balázs Tibor Morvay | CLT8ZP |
| László Kálmán Trautsch | CMLJKQ |
| Marcell Csikós | P78P08 |

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
