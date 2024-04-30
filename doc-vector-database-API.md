# Search API for the vector database

We provide access to a vector database having three fields:

* the id of the document
* embedding vectors computed from the title of the document
* embedding vectors computed from the abstract of the document

Embedding vectors were computed from the Transformer-based [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

## Access

The API is accessible at:

```
https://guacamole.univ-avignon.fr/stvir_test?corpus=abstract&phrase=very long sentence&length=10
```

We recommend to use Firefox to easily visualize json response content.

For each retrieved document, the document itself is provided, with its id and the score used for document ranking. This score is the dot-product of the embedding vectors of the document and the abstract (or title) computed by all-MiniLM-L6-v2.

You can set up the number of documents (<1000) to be retrieved with the parameter length: up to 1000

```
https://guacamole.univ-avignon.fr/stvir_test?corpus=abstract&phrase=very long sentence&length=1000
```

To use this url with wget you can do a command like this:

```
wget -O dblp.json "https://guacamole.univ-avignon.fr/stvir_test?corpus=abstract&phrase=very long query&length=10"
```

## Queries with embeddings

The parameter "corpus" allows users to define full text queries to search inside:

* the abstract of documents, with the value "abstract":

  ```
  https://guacamole.univ-avignon.fr/stvir_test?corpus=abstract&phrase=Exploring the use of AI to improve success rates and speed in the pharmaceutical research field&length=10
  ```

  with encoding:

  ```
  https://guacamole.univ-avignon.fr/stvir_test?corpus=abstract&phrase=Exploring%20the%20use%20of%20AI%20to%20improve%20success%20rates%20and%20speed%20in%20the%20pharmaceutical%20research%20field&length=10
  ```
* the title of documents, with the value "title":

  ```
  https://guacamole.univ-avignon.fr/stvir_test?corpus=title&phrase=Exploring the use of AI to improve success rates and speed in the pharmaceutical research field&length=10
  ```

### Other queries

Use Elastic Search Index to search for a specific field.

Retrieve document with id=1584898773

```
https://guacamole.univ-avignon.fr/dblp1/_search?q=_id:1494645067
```

All documents referencing 1584898773

```
https://guacamole.univ-avignon.fr/dblp1/_search?q=references:1494645067
```

Note that every id document returned by queries done toward the database vector should be inside the Elastic Search Index. The vector database is a subset of this ES index, since we had to remove (nearly) empty abstracts to compute embedding vectors.