docker run -itd --privileged --hostname=78aa085991cf --mac-address=02:42:ac:11:00:02 --env=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --volume=C:\AutoProteus\Docker:/AutoProteus --name ubuntu --restart=no --label='org.opencontainers.image.ref.name=ubuntu' --label='org.opencontainers.image.version=20.04' --runtime=runc --memory="8147483648" -t -d ubuntu:20.04 /bin/bash -c "apt-get update && apt-get install -y sudo && exec bash"