# Search API for DBLP

We provide access to an index for elastic search.

## Access

The API is accessible at:

```
https://guacamole.univ-avignon.fr/dblp1/_search
```

with:

* user: inex
* passwd: qatc2011

We recommend to use Firefox to easily visualize json content.

You can set up the number of documents (<10000) to be retrieved with the parameter size:

```
https://guacamole.univ-avignon.fr/dblp1/_search?size=100
```

To use this url with wget you need to use the parameter --no-check-certificate

```
wget -O dblp.json --no-check-certificate https://inex:qatc2011@guacamole.univ-avignon.fr/dblp1/_search?size=100 
```

## Queries

The parameter q allows users to define full text queries or boolean combinations of field:keyword pairs.

### Example of full text query

All documents mentioning linear algebra:

```
 https://inex:qatc2011@guacamole.univ-avignon.fr/dblp1/_search?q="linear algebra"&size=100 
```

with encoding

```
 https://inex:qatc2011@guacamole.univ-avignon.fr/dblp1/_search?q=%22linear%20algebra%22&size=100 
```

### Examples of boolean query

Retrieve document with id=1584898773

```
https://guacamole.univ-avignon.fr/dblp1/_search?q=_id:1494645067
```

All documents referencing 1584898773

```
https://guacamole.univ-avignon.fr/dblp1/_search?q=references:1494645067
```

Documents by author 1936126667

```
https://guacamole.univ-avignon.fr/dblp1/_search?q=id:1494645067
```

All documents which title contains geometric or with a field of subject name equal to *"Computer science"* and another including *algebra* but none equal to Graph.

```
https://guacamole.univ-avignon.fr/dblp1/_search?q=(( fos.name: "Computer science" AND fos.name:*algebra* AND NOT fos.name:Graph ) OR title:*geometric* )&size=10
```

with encoding:

```
https://guacamole.univ-avignon.fr/dblp1/_search?q=((%20fos.name:%22Computer%20science%22%20AND%20fos.name:*algebra*%20AND%20NOT%20fos.name:Graph%20)%20OR%20title:*geometric*%20)&size=10
```