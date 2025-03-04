1. **Install Anaconda:**

- Download Anaconda from https://docs.anaconda.com/anaconda/install/windows/
- Run the installer and follow the prompts. Note that it takes up several GB and take a while to install, but it will be a powerful platform for you to use in the future.

2. **Set up the environment:**

- Open **Anaconda Prompt** (search for it in the Start menu)
- Create the environment: `conda env create -f environment.yml`
- Wait for a few minutes for all packages to be installed - in some cases, this can literally take 20-30 minutes if you've not used Anaconda before, and even longer depending on your internet connection. Important stuff is happening!
- You have now built an isolated, dedicated AI environment for engineering, running vector datastores, and so much more! You now need to **activate** it using this command: `conda activate ai`

You should see `(ai)` in your prompt, which indicates you've activated your new environment.

3. **Start Jupyter Lab:**

- In the Anaconda Prompt, from within the main folder, type: `jupyter lab`

...and Jupyter Lab should open up in a browser.

4. **Run Jupyter notebook in VS Code**

- Start Jupyter Lab as stated in point 3.
- In VS Code, install `Jupyter` extension pack.
- Open jupyter notebook file, ex. `data_curation_1.ipynb`.
- At the top right corner you should see `Select Kernel` button, click it.
- At the search bar at the top, a dropdown should appear, select `Existing Jupyter Server`.
- From **Anaconda Prompt** copy your local jupyter url, should be similar to `http://localhost:8888/lab?token=7078f5d95f9dc430e4d35b72002c3a8957f04b1783e44929`.
- Paste it into VS Code and press Enter.
- Set Server Display Name or leave it as it is, again press Enter.
- Choose newly created server from a list and you're ready to go! Enjoy!
