#!/usr/bin/env python3
import os

from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def required(requirements_file):
    """ Read requirements file and remove comments and empty lines. """
    with open(os.path.join(BASEDIR, requirements_file), 'r') as f:
        requirements = f.read().splitlines()
        if 'MYCROFT_LOOSE_REQUIREMENTS' in os.environ:
            print('USING LOOSE REQUIREMENTS!')
            requirements = [r.replace('==', '>=').replace('~=', '>=') for r in requirements]
        return [pkg for pkg in requirements
                if pkg.strip() and not pkg.startswith("#")]


with open(os.path.join(BASEDIR, "README.md"), "r") as f:
    long_description = f.read()



def get_version():
    """ Find the version of the package"""
    version = None
    version_file = os.path.join(BASEDIR, 'ovos_solver_openai_persona', 'version.py')
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if 'VERSION_MAJOR' in line:
                major = line.split('=')[1].strip()
            elif 'VERSION_MINOR' in line:
                minor = line.split('=')[1].strip()
            elif 'VERSION_BUILD' in line:
                build = line.split('=')[1].strip()
            elif 'VERSION_ALPHA' in line:
                alpha = line.split('=')[1].strip()

            if ((major and minor and build and alpha) or
                    '# END_VERSION_BLOCK' in line):
                break
    version = f"{major}.{minor}.{build}"
    if alpha and int(alpha) > 0:
        version += f"a{alpha}"
    return version


PERSONA_ENTRY_POINT = 'Remote Llama=ovos_solver_openai_persona:LLAMA_DEMO'
PLUGIN_ENTRY_POINT = 'ovos-solver-openai-plugin=ovos_solver_openai_persona.engines:OpenAICompletionsSolver'
DIALOG_PLUGIN_ENTRY_POINT = 'ovos-dialog-transformer-openai-plugin=ovos_solver_openai_persona.dialog_transformers:OpenAIDialogTransformer'
SUMMARIZER_ENTRY_POINT = 'ovos-summarizer-openai-plugin=ovos_solver_openai_persona.summarizer:OpenAISummarizer'


setup(
    name='ovos-openai-plugin',
    version=get_version(),
    description='A question solver plugin for ovos',
    url='https://github.com/OpenVoiceOS/ovos-solver-plugin-openai-persona',
    author='jarbasai',
    author_email='jarbasai@mailfence.com',
    license='MIT',
    packages=['ovos_solver_openai_persona'],
    zip_safe=True,
    keywords='ovos plugin utterance fallback query',
    entry_points={
        'neon.plugin.solver': PLUGIN_ENTRY_POINT,
        "opm.transformer.dialog": DIALOG_PLUGIN_ENTRY_POINT,
        'opm.solver.summarization': SUMMARIZER_ENTRY_POINT,
        "opm.plugin.persona": PERSONA_ENTRY_POINT
    },
    install_requires=required("requirements.txt"),
    long_description=long_description,
    long_description_content_type='text/markdown'
)
