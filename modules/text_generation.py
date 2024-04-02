import ast
import copy
import html
import pprint
import random
import re
import time
import traceback

import numpy as np
import torch
import transformers
from transformers import LogitsProcessorList, is_torch_xpu_available

import modules.shared as shared
from modules.cache_utils import process_llamacpp_cache
from modules.callbacks import (
    Iteratorize,
    Stream,
    _StopEverythingStoppingCriteria
)
from modules.extensions import apply_extensions
from modules.grammar.grammar_utils import initialize_grammar
from modules.grammar.logits_process import GrammarConstrainedLogitsProcessor
from modules.html_generator import generate_4chan_html, generate_basic_html
from modules.logging_colors import logger
from modules.models import clear_torch_cache, local_rank



import os
import re
import subprocess

def extract_and_save_snippets_to_docker(text, container_name='ubuntu', state_file='/temp/docker_current_dir.txt'):
    # Ensure the state file exists and has a default directory
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    if not os.path.isfile(state_file):
        with open(state_file, 'w') as file:
            file.write('/mnt/data')  # Default directory inside the container
    
    # Initialize the output dictionary
    aggregated_stdout = ''
    aggregated_stderr = ''
    final_exit_code = 0
    
    # Read the current directory from the state file
    with open(state_file, 'r') as file:
        current_dir = file.read().strip() or '/mnt/data'
    
    # Regex to find code snippets
    pattern = r'```(\w+)([\s\S]+?)```'
    matches = re.findall(pattern, text, re.MULTILINE)

    language_extensions = {'cpp': 'cpp',
        'c': 'c',
        'h': 'h',
        'py': 'py',
        'python': 'py',
        'js': 'js',
        'javascript': 'js',
        'java': 'java',
        'html': 'html',
        'htm': 'html',
        'css': 'css',
        'cs': 'cs',
        'csharp': 'cs',
        'hlsl': 'hlsl',
        'rs': 'rs',
        'rust': 'rs',
        'go': 'go',
        'rb': 'rb',
        'ruby': 'rb',
        'php': 'php',
        'ts': 'ts',
        'typescript': 'ts',
        'sh': 'sh',
        'bash': 'sh',
        'lua': 'lua',
        'pl': 'pl',
        'perl': 'pl',
        'scala': 'scala',
        'swift': 'swift',
        'kt': 'kt',
        'kotlin': 'kt',
        'sql': 'sql',
        'r': 'r',
        'yaml': 'yaml',
        'yml': 'yaml',
        'xml': 'xml',
        'md': 'md',
        'markdown': 'md',
        'json': 'json',
        'ini': 'ini',
        'csv': 'csv',
        'shell': 'sh'}

    for language, code in matches:
        try:
            code = code.strip()
            extension = language_extensions.get(language, 'txt')
            # Adjusting filename generation to avoid starting with underscores
            base_filename = f"snippet_{re.sub('[^a-zA-Z0-9_]', '_', code.split()[0].lower())}"
            if base_filename.startswith('_'):
                base_filename = "snippet" + base_filename
            filename = f"{base_filename}.{extension}"
            temp_filepath = os.path.join("/tmp", filename)  # Constructing the full temp file path
            docker_filepath = os.path.join(current_dir, filename)
    
            # Write the snippet to the temporary file, ensuring directory exists
            os.makedirs(os.path.dirname(temp_filepath), exist_ok=True)
            with open(temp_filepath, 'w') as temp_file:
                temp_file.write(code)
    
            # Copy the temporary file to the Docker container
            copy_command = f"docker cp {temp_filepath} {container_name}:{docker_filepath}"
            result = subprocess.run(copy_command, shell=True, capture_output=True, text=True, check=False)
    
            if result.returncode == 0:
                aggregated_stdout += f"\n{filename} saved to {docker_filepath}"
            else:
                aggregated_stderr += f"\nError saving {filename} to {docker_filepath}: {result.stderr}"
                final_exit_code = result.returncode
        except Exception as e:
            aggregated_stderr += f"\nError processing {filename}: {str(e)}"
            final_exit_code = 1



    output = f'snippet stdout:\n{{{aggregated_stdout}}}\n'
    #snippet stderr:\n{{{aggregated_stderr}}}\nsnippet exit_code:\n{{{final_exit_code}}}\n'
    return output

   
def execute_docker_command(commands, container_name='ubuntu', state_file='/temp/docker_current_dir.txt', top_mode='once'):
    # Ensure the state file exists and has a default directory
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    if not os.path.isfile(state_file):
        with open(state_file, 'w') as file:
            file.write('/')
    
    # Initialize the output dictionary
    aggregated_stdout = ''
    aggregated_stderr = ''
    final_exit_code = 0
    
    # Read the current directory from the state file
    with open(state_file, 'r') as file:
        current_dir = file.read().strip() or '/'
    
    # Split the commands by '&&', handling quoted strings
    command_list = []
    command = ''
    in_quote = False
    quote_char = ''
    for char in commands:
        if char in "'\"" and (not in_quote or quote_char == char):
            in_quote = not in_quote
            quote_char = char if in_quote else ''
        elif char == '&&' and not in_quote:
            command_list.append(command.strip())
            command = ''
            continue
        command += char
    command_list.append(command.strip())  # Add the last command
    
    # Determine the shell to use in the container
    shell = '/bin/sh'
    test_shell_command = f'docker exec {container_name} {shell} -c "echo shell_test"'
    shell_test_result = subprocess.run(test_shell_command, shell=True, capture_output=True, text=True, check=False)
    if shell_test_result.returncode != 0:
        shell = '/bin/bash'  # Fallback to bash if sh is not available
    
    for cmd in command_list:
        if not cmd:
            continue
        
        # Modify the top command based on top_mode
        if cmd.startswith('top') and top_mode == 'once':
            cmd += ' -n 1'  # Only one iteration
        
        # Execute command
        full_command = f'cd {current_dir} && {cmd}'
        result = subprocess.run(['docker', 'exec', '-it', container_name, shell, '-c', full_command],
                                capture_output=True, text=True, check=False)
        aggregated_stdout += result.stdout
        aggregated_stderr += result.stderr
        final_exit_code = result.returncode

        # Update the current directory if 'cd' is in the command
        if cmd.startswith('cd '):
            current_dir_command = f'docker exec -it {container_name} {shell} -c "pwd"'
            pwd_result = subprocess.run(current_dir_command, shell=True, capture_output=True, text=True, check=False)
            if pwd_result.returncode == 0 and pwd_result.stdout:
                current_dir = pwd_result.stdout.strip()
                with open(state_file, 'w') as file:
                    file.write(current_dir)
    
    # Formatting the output
    output = f'stdout:\n{{{aggregated_stdout}}}\nstderr:\n{{{aggregated_stderr}}}\nexit_code:\n{{{final_exit_code}}}\n'
    
    return output







def _generate_reply(question, state, stopping_strings=None, is_chat=False, escape_html=False, for_ui=False):

    # Find the appropriate generation function
    generate_func = apply_extensions('custom_generate_reply')
    if generate_func is None:
        if shared.model_name == 'None' or shared.model is None:
            logger.error("No model is loaded! Select one in the Model tab.")
            yield ''
            return

        if shared.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'CtransformersModel']:
            generate_func = generate_reply_custom
        else:
            generate_func = generate_reply_HF

    if generate_func != generate_reply_HF and shared.args.verbose:
        logger.info("PROMPT=")
        print(question)
        print()

    # Prepare the input
    original_question = question
    if not is_chat:
        state = apply_extensions('state', state)
        question = apply_extensions('input', question, state)

    # Find the stopping strings
    all_stop_strings = []
    for st in (stopping_strings, state['custom_stopping_strings']):
        if type(st) is str:
            st = ast.literal_eval(f"[{st}]")

        if type(st) is list and len(st) > 0:
            all_stop_strings += st

    shared.stop_everything = False
    clear_torch_cache()
    seed = set_manual_seed(state['seed'])
    last_update = -1
    reply = ''
    is_stream = state['stream']
    if len(all_stop_strings) > 0 and not state['stream']:
        state = copy.deepcopy(state)
        state['stream'] = True

    min_update_interval = 0
    if state.get('max_updates_second', 0) > 0:
        min_update_interval = 1 / state['max_updates_second']

    # Generate
    for reply in generate_func(question, original_question, seed, state, stopping_strings, is_chat=is_chat):
        reply, stop_found = apply_stopping_strings(reply, all_stop_strings)
        if escape_html:
            reply = html.escape(reply)
        if is_stream:
            cur_time = time.time()

            # Maximum number of tokens/second
            if state['max_tokens_second'] > 0:
                diff = 1 / state['max_tokens_second'] - (cur_time - last_update)
                if diff > 0:
                    time.sleep(diff)

                last_update = time.time()
                yield reply

            # Limit updates to avoid lag in the Gradio UI
            # API updates are not limited
            else:
                if cur_time - last_update > min_update_interval:
                    last_update = cur_time
                    yield reply

        if stop_found or (state['max_tokens_second'] > 0 and shared.stop_everything):
            break

    if not is_chat:
        reply = apply_extensions('output', reply, state)

    yield reply



def generate_reply(*args, **kwargs):
    shared.generation_lock.acquire()
    try:
        for result in _generate_reply(*args, **kwargs):
            yield result
    finally:
        shared.generation_lock.release()


def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    if shared.tokenizer is None:
        raise ValueError('No tokenizer is loaded')

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'CtransformersModel', 'Exllamav2Model']:
        input_ids = shared.tokenizer.encode(str(prompt))
        if shared.model.__class__.__name__ not in ['Exllamav2Model']:
            input_ids = np.array(input_ids).reshape(1, len(input_ids))
    else:
        input_ids = shared.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=add_special_tokens)
        if not add_bos_token:
            while len(input_ids[0]) > 0 and input_ids[0][0] == shared.tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]

    # Handling truncation
    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'CtransformersModel'] or shared.args.cpu:
        return input_ids
    elif shared.args.deepspeed:
        return input_ids.to(device=local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        return input_ids.to(device)
    elif is_torch_xpu_available():
        return input_ids.to("xpu:0")
    else:
        return input_ids.cuda()


def decode(output_ids, skip_special_tokens=True):
    if shared.tokenizer is None:
        raise ValueError('No tokenizer is loaded')

    return shared.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)


def get_encoded_length(prompt):
    length_after_extensions = apply_extensions('tokenized_length', prompt)
    if length_after_extensions is not None:
        return length_after_extensions

    return len(encode(prompt)[0])


def get_token_ids(prompt):
    tokens = encode(prompt)[0]
    decoded_tokens = [shared.tokenizer.decode([i]) for i in tokens]

    output = ''
    for row in list(zip(tokens, decoded_tokens)):
        output += f"{str(int(row[0])).ljust(5)}  -  {repr(row[1])}\n"

    return output


def get_max_prompt_length(state):
    return state['truncation_length'] - state['max_new_tokens']


def generate_reply_wrapper(question, state, stopping_strings=None):
    """
    Returns formatted outputs for the UI
    """
    reply = question if not shared.is_seq2seq else ''
    yield formatted_outputs(reply, shared.model_name)

    for reply in generate_reply(question, state, stopping_strings, is_chat=False, escape_html=True, for_ui=True):
        if not shared.is_seq2seq:
            reply = question + reply

        yield formatted_outputs(reply, shared.model_name)


def formatted_outputs(reply, model_name):
    if any(s in model_name for s in ['gpt-4chan', 'gpt4chan']):
        reply = fix_gpt4chan(reply)
        return html.unescape(reply), generate_4chan_html(reply)
    else:
        return html.unescape(reply), generate_basic_html(reply)


def fix_gpt4chan(s):
    """
    Removes empty replies from gpt4chan outputs
    """
    for i in range(10):
        s = re.sub("--- [0-9]*\n>>[0-9]*\n---", "---", s)
        s = re.sub("--- [0-9]*\n *\n---", "---", s)
        s = re.sub("--- [0-9]*\n\n\n---", "---", s)

    return s


def fix_galactica(s):
    """
    Fix the LaTeX equations in GALACTICA
    """
    s = s.replace(r'\[', r'$')
    s = s.replace(r'\]', r'$')
    s = s.replace(r'\(', r'$')
    s = s.replace(r'\)', r'$')
    s = s.replace(r'$$', r'$')
    s = re.sub(r'\n', r'\n\n', s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif is_torch_xpu_available():
        torch.xpu.manual_seed_all(seed)

    return seed


def stop_everything_event():
    shared.stop_everything = True


def apply_stopping_strings(reply, all_stop_strings):
    stop_found = False

    tool_call_prefix = '<tool_call>'
    
    for string in all_stop_strings:
        idx = reply.find(string)
        if idx != -1:
            # Check if the reply contains a tool call before the stopping string
            tool_call_idx = reply.find(tool_call_prefix)
            if tool_call_idx != -1 and tool_call_idx < idx:
                # Extract and execute the command
                command_start = tool_call_idx + len(tool_call_prefix)
                command_end = idx if idx != -1 else len(reply)
                command = reply[command_start:command_end] #.strip()

                snippet_response = extract_and_save_snippets_to_docker(reply)
                wsl_response = execute_docker_command(command)
                
                reply = reply + f'<|im_end|> \n___\n <|im_start|>terminal \n<tool_response>\n {wsl_response}\n{snippet_response}\n</tool_response><|im_end|> \n___\n'


            else:
                reply = reply[:idx]


            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        for string in all_stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found


def get_reply_from_output_ids(output_ids, state=None, starting_from=0):
    reply = decode(output_ids[starting_from:], state['skip_special_tokens'] if state else True)

    # Handle tokenizers that do not add the leading space for the first token
    if (hasattr(shared.tokenizer, 'convert_ids_to_tokens') and len(output_ids) > starting_from) and not reply.startswith(' '):
        first_token = shared.tokenizer.convert_ids_to_tokens(int(output_ids[starting_from]))
        if isinstance(first_token, (bytes,)):
            first_token = first_token.decode('utf8')

        if first_token.startswith('▁'):
            reply = ' ' + reply

    return reply


def generate_reply_HF(question, original_question, seed, state, stopping_strings=None, is_chat=False):
    generate_params = {}
    for k in ['max_new_tokens', 'temperature', 'temperature_last', 'dynamic_temperature', 'dynatemp_low', 'dynatemp_high', 'dynatemp_exponent', 'smoothing_factor', 'smoothing_curve', 'top_p', 'min_p', 'top_k', 'repetition_penalty', 'presence_penalty', 'frequency_penalty', 'repetition_penalty_range', 'typical_p', 'tfs', 'top_a', 'guidance_scale', 'penalty_alpha', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'do_sample', 'encoder_repetition_penalty', 'no_repeat_ngram_size', 'min_length', 'num_beams', 'length_penalty', 'early_stopping']:
        if k in state:
            generate_params[k] = state[k]

    if isinstance(state['sampler_priority'], list) and len(state['sampler_priority']) > 0:
        generate_params['sampler_priority'] = state['sampler_priority']
    elif isinstance(state['sampler_priority'], str) and state['sampler_priority'].strip() != '':
        generate_params['sampler_priority'] = [x.strip() for x in state['sampler_priority'].replace('\n', ',').split(',') if x.strip()]

    if state['negative_prompt'] != '':
        generate_params['negative_prompt_ids'] = encode(state['negative_prompt'])

    if state['prompt_lookup_num_tokens'] > 0:
        generate_params['prompt_lookup_num_tokens'] = state['prompt_lookup_num_tokens']

    for k in ['epsilon_cutoff', 'eta_cutoff']:
        if state[k] > 0:
            generate_params[k] = state[k] * 1e-4

    if state['ban_eos_token']:
        generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    if state['custom_token_bans']:
        to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
        if len(to_ban) > 0:
            if generate_params.get('suppress_tokens', None):
                generate_params['suppress_tokens'] += to_ban
            else:
                generate_params['suppress_tokens'] = to_ban

    generate_params.update({'use_cache': not shared.args.no_cache})
    if shared.args.deepspeed:
        generate_params.update({'synced_gpus': True})

    # Encode the input
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    output = input_ids[0]
    cuda = not any((shared.args.cpu, shared.args.deepspeed))
    if state['auto_max_new_tokens']:
        generate_params['max_new_tokens'] = state['truncation_length'] - input_ids.shape[-1]

    # Add the encoded tokens to generate_params
    question, input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, input_ids, None)
    original_input_ids = input_ids
    generate_params.update({'inputs': input_ids})
    if inputs_embeds is not None:
        generate_params.update({'inputs_embeds': inputs_embeds})

    # Stopping criteria / eos token
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    generate_params['eos_token_id'] = eos_token_ids
    generate_params['stopping_criteria'] = transformers.StoppingCriteriaList()
    generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria())

    # Logits processor
    processor = state.get('logits_processor', LogitsProcessorList([]))
    if not isinstance(processor, LogitsProcessorList):
        processor = LogitsProcessorList([processor])

    # Grammar
    if state['grammar_string'].strip() != '':
        grammar = initialize_grammar(state['grammar_string'])
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
        processor.append(grammar_processor)

    apply_extensions('logits_processor', processor, input_ids)
    generate_params['logits_processor'] = processor

    if shared.args.verbose:
        logger.info("GENERATE_PARAMS=")
        filtered_params = {key: value for key, value in generate_params.items() if not isinstance(value, torch.Tensor)}
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(filtered_params)
        print()

        logger.info("PROMPT=")
        print(decode(input_ids[0], skip_special_tokens=False))
        print()

    # Handle StreamingLLM for llamacpp_HF
    if shared.model.__class__.__name__ == 'LlamacppHF' and shared.args.streaming_llm:
        tmp = process_llamacpp_cache(shared.model.model, input_ids[-1].tolist(), shared.model.model._input_ids.tolist())
        shared.model.past_seq = torch.tensor(tmp)
        shared.model.save_cache()

    t0 = time.time()
    try:
        if not is_chat and not shared.is_seq2seq:
            yield ''

        # Generate the entire reply at once.
        if not state['stream']:
            with torch.no_grad():
                output = shared.model.generate(**generate_params)[0]
                if cuda:
                    output = output.cuda()

            starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
            yield get_reply_from_output_ids(output, state, starting_from=starting_from)

        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator.
        else:

            def generate_with_callback(callback=None, *args, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                clear_torch_cache()
                with torch.no_grad():
                    shared.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, [], kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                cumulative_reply = ''
                starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
                for output in generator:
                    if output[-1] in eos_token_ids:
                        break

                    new_content = get_reply_from_output_ids(output, state, starting_from=starting_from)
                    # check the partial unicode character
                    if chr(0xfffd) in new_content:
                        continue

                    cumulative_reply += new_content
                    starting_from = len(output)
                    yield cumulative_reply

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = len(output) - (original_tokens if not shared.is_seq2seq else 0)
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return


def generate_reply_custom(question, original_question, seed, state, stopping_strings=None, is_chat=False):
    """
    For models that do not use the transformers library for sampling
    """
    seed = set_manual_seed(state['seed'])

    t0 = time.time()
    reply = ''
    try:
        if not is_chat:
            yield ''

        if not state['stream']:
            reply = shared.model.generate(question, state)
            yield reply
        else:
            for reply in shared.model.generate_with_streaming(question, state):
                yield reply

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(encode(original_question)[0])
        new_tokens = len(encode(original_question + reply)[0]) - original_tokens
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return
