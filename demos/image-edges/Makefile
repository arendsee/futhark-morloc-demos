.PHONY: all clean

all:
	morloc make -o nexus -X futhark:backend=multicore main.loc
	./nexus edges sample.png edges.png

clean:
	rm -rf nexus nexus-build pools __pycache__ *.err edges.png
