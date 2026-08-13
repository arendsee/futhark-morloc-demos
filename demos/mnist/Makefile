.PHONY: all clean

all:
	morloc make -o nexus main.loc
	./nexus train 150 mnist.png

clean:
	rm -rf nexus pools __pycache__ *.err mnist.png out
