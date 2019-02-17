import argparse
import unique_faces as uf
import FaceAPI as fa

def parseArgs():
    """Parse command line arguments and return args dict"""
    parser = argparse.ArgumentParser(description='Summarize annovar annotated vcfs')
    parser.add_argument('-I', '--input', type=str, nargs=1, dest='input_file')
    parser.add_argument('-O', '--output', type=str, nargs=1, dest='output_file')
    args = parser.parse_args()

    arguments = {}
    if args.input_file is None:
        raise ValueError("No input file specified!")
    input_file = ''.join(args.input_file)
    arguments['input_file'] = input_file
    if args.output_file is None:
        arguments['output_file'] = '{}_identified'.format(input_file)
    else:
        arguments['output_file'] = ''.join(args.output_file)

    return arguments

def main():
    args = parseArgs()
    #run ketan's code here to give me the embeddings and csv data
    fa.setup_CF()
    fa.load_model()
    frame_data, embeddings = fa.annotate_vid(args['input_file'])
    #associate person labels with each frame
    uf.label_persons(embeddings, frame_data, args['output_file'])

    


if __name__ == '__main__':
    main()

