FROM rlincoln/oct2pypower

RUN pip3 install texttable cvxopt

ADD . /usr/src/andes
#RUN cd /usr/src/andes && python3 setup.py install
ENV PYTHONPATH /usr/src/andes

ENTRYPOINT ["python3", "-m", "andes.test.test_matpower"]
