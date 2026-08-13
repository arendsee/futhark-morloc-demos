.PHONY: all clean

all:
	morloc make -o galaxy -X futhark:backend=multicore main.loc
	./galaxy simulate 400 60 out

clean:
	rm -rf nexus *-build __pycache__ *.err out
