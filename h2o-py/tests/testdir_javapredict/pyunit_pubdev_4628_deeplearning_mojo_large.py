from builtins import range
import sys, os
sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
import random
from random import randint
import re
import subprocess
from subprocess import STDOUT,PIPE
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

NTESTROWS = 1000    # number of test dataset rows
MAXLAYERS = 6
MAXNODESPERLAYER = 20
TMPDIR = ""
POJONAME = ""

def deeplearning_mojo():
    allAct = ["maxout", "rectifier", "maxout_with_dropout", "tanh_with_dropout", "rectifier_with_dropout", "tanh"]
    problemType = ["binomial", "multinomial", "regression"]
    missingValues = ['Skip', 'MeanImputation']
    allFactors = [True, False]
    categoricalEncodings = ['auto', 'one_hot_internal', 'binary', 'eigen']

    problem = problemType[randint(0,len(problemType)-1)]
    actFunc = allAct[randint(0,len(allAct)-1)]
    missing_values = missingValues[randint(0, len(missingValues)-1)]
    cateEn = categoricalEncodings[randint(0, len(categoricalEncodings)-1)]
    toStandardize = allFactors[randint(0, len(allFactors) - 1)]
    useAllFactors = allFactors[randint(0, len(allFactors) - 1)]

    for problem in problemType:
        if (problem == 'regression'):
            loss = 'Automatic'
            prob = 'numeric'
        else:
            loss = 'CrossEntropy'
            prob = 'class'

        for actFunc in allAct:
            for missing_values in missingValues:
                for cateEn in categoricalEncodings:
                    for toStandardize in allFactors:
                        for useAllFactors in allFactors:
                            hiddens, hidden_dropout_ratios = random_networkSize(actFunc)  # generate random size layers
                            params = {}
                            if ('dropout') in actFunc:
                                params = {'loss': loss, 'hidden': hiddens, 'standardize': toStandardize,
                                          'missing_values_handling': missing_values, 'activation': actFunc,
                                          'use_all_factor_levels': useAllFactors,
                                          'hidden_dropout_ratios': hidden_dropout_ratios,
                                          'input_dropout_ratio': random.uniform(0, 0.5),
                                          'categorical_encoding': cateEn
                                          }
                            else:
                                params = {'loss': loss, 'hidden': hiddens, 'standardize': toStandardize,
                                          'missing_values_handling': missing_values, 'activation': actFunc,
                                          'use_all_factor_levels': useAllFactors,
                                          'input_dropout_ratio': random.uniform(0, 0.5),
                                          'categorical_encoding': cateEn
                                          }
                            print("**********  Starting new test...")
                            print(params)
                            df = random_dataset(problem)  # generate random dataset
                            train = df[NTESTROWS:, :]
                            test = df[:NTESTROWS, :]
                            x = list(set(df.names) - {"response"})

                            try:
                                # build a model
                                build_save_model(params, x, train)
                                h2o.download_csv(test[x], os.path.join(TMPDIR, 'in.csv'))  # save test file, h2o predict/mojo use same file
                                # load model and perform predict
                                pred_h2o = mojo_predict(x)

                                # load prediction into a frame and compare
                                pred_mojo = h2o.import_file(os.path.join(TMPDIR, 'out_mojo.csv'))
                                pyunit_utils.compare_frames(pred_h2o, pred_mojo, min(1000, pred_mojo.ncols*pred_mojo.nrows), tol_numeric=1e-4)
                            except Exception as ex:
                                if hasattr(ex, 'args') and type(ex.args[0]==type("what")):
                                    if "unstable model" not in ex.args[0]:
                                        print(params)
                                        print(ex)
                                        sys.exit(1)     # okay to encounter unstable model not nothingh else
                                    else:
                                        print("An unstable model is found and no mojo is built.")

# perform h2o predict and mojo predict.  Frame containing h2o prediction is returned and mojo predict is
# written to file.
def mojo_predict(x):
    newTest = h2o.import_file(os.path.join(TMPDIR, 'in.csv'))   # Make sure h2o and mojo use same in.csv
    newModel = h2o.load_model(path=os.path.join(TMPDIR, POJONAME)) # perform h2o predict
    predictions1 = newModel.predict(newTest)

    # load mojo and have it do predict
    outFileName = os.path.join(TMPDIR, 'out_mojo.csv')
    java_cmd = ["java", "-ea", "-cp", "/Users/wendycwong/h2o-3/h2o-assemblies/genmodel/build/libs/genmodel.jar",
                "-Xmx12g", "-XX:MaxPermSize=2g", "-XX:ReservedCodeCacheSize=256m", "hex.genmodel.tools.PredictCsv",
                "--input", os.path.join(TMPDIR, 'in.csv'), "--output",
                outFileName, "--mojo", os.path.join(TMPDIR, POJONAME)+".zip", "--decimal"]
    p = subprocess.Popen(java_cmd, stdout=PIPE, stderr=STDOUT)
    o, e = p.communicate()
    return predictions1

def build_save_model(params, x, train):
    global TMPDIR
    global POJONAME
    # build a model
    model = H2ODeepLearningEstimator(**params)
    if params['autoencoder']:
        model.train(x=x, training_frame=train)
    else:
        model.train(x=x, y="response", training_frame=train)
    # save model
    regex = re.compile("[+\\-* !@#$%^&()={}\\[\\]|;:'\"<>,.?/]")
    POJONAME = regex.sub("_", model._id)

    print("Downloading Java prediction model code from H2O")
    TMPDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "results", POJONAME))
    os.makedirs(TMPDIR)
    h2o.save_model(model, path=TMPDIR, force=True)  # save h2o model
    model.download_mojo(path=TMPDIR)    # save mojo
    h2o.remove(model)

# generate random neural network architecture
def random_networkSize(actFunc):
    no_hidden_layers = randint(1, MAXLAYERS)
    hidden = []
    hidden_dropouts = []
    for k in range(1, no_hidden_layers+1):
        hidden.append(randint(1,MAXNODESPERLAYER))
        if 'dropout' in actFunc.lower():
            hidden_dropouts.append(random.uniform(0,0.5))

    return hidden, hidden_dropouts

# generate random dataset
def random_dataset(response_type, verbose=True):
    """Create and return a random dataset."""
    if verbose: print("\nCreating a dataset for a %s problem:" % response_type)
    fractions = {k + "_fraction": random.random() for k in "real categorical integer time string binary".split()}
    fractions["string_fraction"] = 0  # Right now we are dropping string columns, so no point in having them.
    fractions["binary_fraction"] /= 3
    fractions["time_fraction"] /= 2
    # fractions["categorical_fraction"] = 0
    sum_fractions = sum(fractions.values())
    for k in fractions:
        fractions[k] /= sum_fractions
    response_factors = (1 if response_type == "regression" else
                        2 if response_type == "binomial" else
                        random.randint(3, 10))
    df = h2o.create_frame(rows=random.randint(15000, 25000) + NTESTROWS, cols=random.randint(20, 100),
                          missing_fraction=random.uniform(0, 0.05),
                          has_response=True, response_factors=response_factors, positive_response=True,
                          **fractions)
    if verbose:
        print()
        df.show()
    return df

if __name__ == "__main__":
    pyunit_utils.standalone_test(deeplearning_mojo)
else:
    deeplearning_mojo()
