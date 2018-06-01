#include "MaskGenerating.h"

int main() 
{
	try {

		// MASK GENERATING

		eiid::generateMasksDataset(std::string(PATH));

		return 1;

	}
	catch (aia::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}

