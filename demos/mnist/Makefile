.PHONY: all clean

all:
	morloc make -o nexus -X futhark:backend=multicore main.loc
	./nexus train 150 mnist.png > weights.json

clean:
	rm -rf nexus pools __pycache__ *.err mnist.png out
