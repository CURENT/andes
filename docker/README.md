Dockerfile in this directory downloads ANDES from pip and builds it into a Docker image.

Build with

```bash
docker build . -t andes:latest
```

One can create an alias in shell

```bash
alias andesd='docker run -v `pwd`:/andes cuihantao/andes:latest'
```
and then use `andesd` just in place of `andes`.

See more details at https://cui.eecps.com/blog/2020/andes-docker/