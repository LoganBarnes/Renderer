#include <iostream>
#include <string>
#include <stdexcept>
#include <limits>
#include "renderer/renderer.hpp"


const uint MAX_NUMBER = std::numeric_limits<unsigned short>::max();

inline unsigned int stoui(const std::string& s)
{
	unsigned long lresult = std::stoul(s, nullptr, 10);
	unsigned short result = lresult;
	if (result != lresult) throw std::out_of_range("");
	return result;
}


void error_callback(int error, const char* description)
{
    std::cerr << description << std::endl;
}



bool evaluate(std::string str, unsigned int& num)
{
	try
	{
		num = stoui (str);
		std::cout << "Using: " << num << " as the number." << std::endl;
		return true;
	}
	catch (const std::invalid_argument& ia)
	{
		std::cout << "ERROR: " << str << " is not an integer." << std::endl;
	}
	catch (const std::out_of_range& oor)
	{
		std::cout << "ERROR: " << str << " is not between 0 and " << MAX_NUMBER << "." << std::endl;
	}
	return false;
}


int main(int argc, const char **argv)
{
	Renderer *renderer = new Renderer(argc, argv, MAX_NUMBER);
	if (!renderer->initGLFW(error_callback))
	{
		std::cerr << "GLFW failed to initialize. Exiting program..." << std::endl;
		return 1;
	}

	unsigned int num;
	unsigned long result;

	if (argc > 1)
	{
		if (!strcmp(argv[1], "-s")) // evaluate input once and exit
		{
			if (argc > 2)
			{
				if (evaluate(argv[2], num))
				{
					result = renderer->sumNumber(num);
					std::cout << "The sum from 0 to " << num << " is " << result << std::endl;
				}
			}
			return 0;
		}
	}
	std::cout << "Type 'q' 'quit' or 'exit' to terminate the program." << std::endl;;

	std::string str = "";

	std::cout << "\nEnter an integer from 0 to " << MAX_NUMBER << ": ";
	std::getline (std::cin, str);

	while (str.compare("q") && str.compare("quit") && str.compare("exit"))
	{
		if (evaluate(str, num))
		{
			result = renderer->sumNumber(num);
			std::cout << "The sum from 0 to " << num << " is " << result << std::endl;
		}

		std::cout << "\nEnter an integer from 0 to " << MAX_NUMBER << ": ";
		std::getline (std::cin, str);
	}

	renderer->terminateGLFW();
	delete renderer;

	std::cout << "exiting" << std::endl;

	return 0;
}
