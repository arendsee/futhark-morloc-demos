.PHONY: all clean

all:
	morloc make -o nexus -X futhark:backend=multicore main.loc
	./nexus cluster 300 kmeans.png

clean:
	rm -rf nexus pools __pycache__ *.err kmeans.png
